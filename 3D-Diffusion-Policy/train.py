if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.profiler
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import time
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.policy.gsplat_dp3 import GSplatDP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.data_augmentations import GaussianCompose
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler


OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    # torch.compile now wraps submodules inside each policy (off the module registry),
    # so no compiled wrapper ever lands in a saved state_dict — nothing to exclude.
    exclude_keys = ()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        if cfg.policy._target_ == "diffusion_policy_3d.policy.dp3.DP3":
            self.model: DP3 = hydra.utils.instantiate(cfg.policy)
        elif cfg.policy._target_ == "diffusion_policy_3d.policy.gsplat_dp3.GSplatDP3":
            self.model: GSplatDP3 = hydra.utils.instantiate(cfg.policy)
        elif cfg.policy._target_ == "diffusion_policy_3d.policy.mindmap_dp3.MindmapDP3":
            # from diffusion_policy_3d.policy.mindmap_dp3 import MindmapDP3
            # self.model: MindmapDP3 = hydra.utils.instantiate(cfg.policy)
            raise ValueError(f"Integration not yet finished for {cfg.policy._target_}")
        else:
            raise ValueError(f"Unknown policy target: {cfg.policy._target_}")

        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except Exception: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        # configure training state
        # Policies may define their own parameter grouping (e.g. MindmapDP3
        # replicates mindmap's no-weight-decay-for-bias/LayerNorm groups).
        # Constructed directly (not via hydra.instantiate) because hydra wraps
        # the param-group dicts into DictConfigs, which torch optimizers reject.
        if hasattr(self.model, "get_optimizer_param_groups"):
            param_groups = self.model.get_optimizer_param_groups(cfg.optimizer.weight_decay)
            optimizer_cls = hydra.utils.get_class(cfg.optimizer._target_)
            optimizer_kwargs = {
                k: v for k, v in OmegaConf.to_container(cfg.optimizer, resolve=True).items()
                if k != "_target_"
            }
            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        else:
            # same hyperparameters for all parameters
            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 3
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 2
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            verbose = True
            cfg.task.env_runner.eval_episodes = 2
            cfg.task.dataset.max_train_episodes = 5
        else:
            RUN_ROLLOUT = True
            verbose = False
        
        RUN_VALIDATION = True # reduce time cost
        
        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseDataset), f"dataset must be BaseDataset, got {type(dataset)}"
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # configure lr scheduler
        # Stored on self so save_checkpoint/load_payload pick it up via its
        # state_dict. Constructed fresh (last_epoch=-1) BEFORE the resume block:
        # torch's recommended resume is load_state_dict on a fresh scheduler,
        # which is exact for both branches.
        # All schedulers here are stepped every optimizer update, not every epoch.
        steps_per_epoch = len(train_dataloader) // cfg.training.gradient_accumulate_every

        if cfg.training.lr_scheduler == "linear_to_end_factor":
            # mindmap's schedule: linear decay from lr to lr*end_factor over the
            # first convergence_percentage of training steps, constant afterwards.
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=cfg.training.lr_start_factor,
                end_factor=cfg.training.lr_end_factor,
                total_iters=int(
                    steps_per_epoch * cfg.training.num_epochs
                    * cfg.training.lr_convergence_percentage
                ),
            )
        else:
            self.lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=steps_per_epoch * cfg.training.num_epochs,
            )

        # resume training (restores model, ema_model, optimizer and lr_scheduler
        # states plus global_step/epoch)
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                # Checkpoint is saved at the end of an epoch before epoch is incremented.
                # global_step (optimizer updates) is already at its correct value.
                self.epoch += 1

        # configure GPU augmentations
        if 'train_data_augmentations' in cfg.task and cfg.task.train_data_augmentations is not None:
            augmentations = [hydra.utils.instantiate(t_cfg) for t_cfg in cfg.task.train_data_augmentations]
            self.train_data_augmentations = GaussianCompose(augmentations)
        else:
            raise ValueError("Train data augmentations must be explicitly provided in the dataset config!")
            
        if 'val_data_augmentations' in cfg.task and cfg.task.val_data_augmentations is not None:
            augmentations = [hydra.utils.instantiate(t_cfg) for t_cfg in cfg.task.val_data_augmentations]
            self.val_data_augmentations = GaussianCompose(augmentations)
        else:
            raise ValueError("Eval data augmentations must be explicitly provided in the dataset config!")

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Data-derived normalization bounds (nvblox/mindmap datasets expose them
        # from the zarr attr). Set on BOTH models: EMA only averages parameters,
        # not buffers. Persisted via state_dict, so checkpoints carry them.
        if hasattr(dataset, 'workspace_bounds'):
            self.model.set_workspace_bounds(dataset.workspace_bounds)
            if cfg.training.use_ema:
                self.ema_model.set_workspace_bounds(dataset.workspace_bounds)

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            ema.optimization_step = self.global_step

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)

        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
                "train_episode_indices": np.nonzero(dataset.global_train_mask)[0].tolist(), 
                "val_episode_indices": np.nonzero(val_dataset.global_train_mask)[0].tolist(), 
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        
        # operator fusion via torch.compile (opt out for models with data-dependent
        # loops that trace poorly, e.g. DiffuserActor's FPS). Each policy compiles its
        # own compute-heavy submodules in place while keeping the uncompiled originals
        # registered, so checkpoints stay portable (see BasePolicy.apply_torch_compile).
        if cfg.training.use_torch_compile:
            compile_mode = cfg.training.get('torch_compile_mode', 'default')
            self.model.apply_torch_compile(compile_mode)
            if self.ema_model is not None:
                self.ema_model.apply_torch_compile(compile_mode)

        # pre-select a fixed random batch for action MSE tracking (never used for gradient steps)
        # augmentations applied once here so the metric is fully deterministic across epochs
        sample_indices = torch.randint(len(dataset), (cfg.dataloader.batch_size,)).tolist()
        train_sampling_batch = torch.utils.data.default_collate([dataset[i] for i in sample_indices])
        train_sampling_batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
        train_sampling_batch = self.train_data_augmentations(train_sampling_batch)

        # training loop
        for _ in range(self.epoch, cfg.training.num_epochs):
            epoch_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            self.optimizer.zero_grad(set_to_none=True)
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                t_before_next_batch = time.time()
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()

                    # device transfer.
                    # The copy is async (non_blocking=True), so CPU wall-clock (t1_1 - t1) only
                    # measures the LAUNCH, not the copy — and the copy's real cost would otherwise
                    # leak into the next op's timing. We measure the true GPU transfer time with
                    # CUDA events (recorded on the stream, no CPU stall); we read elapsed_time later
                    # inside `if verbose` after a natural sync (loss.backward / .item), so NO extra
                    # synchronize is added to the hot path.
                    if verbose and torch.cuda.is_available():
                        h2d_start = torch.cuda.Event(enable_timing=True)
                        h2d_end = torch.cuda.Event(enable_timing=True)
                        h2d_start.record()
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if verbose and torch.cuda.is_available():
                        h2d_end.record()
                    t1_1 = time.time()

                    # augmentations
                    batch = self.train_data_augmentations(batch)
                    t1_2 = time.time()
                    
                    # compute loss
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    t1_3 = time.time()

                    # step optimizer
                    t1_4 = t1_3
                    if (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                        # gradient clipping & norm computation (must happen before optimizer.step)
                        if cfg.training.grad_clip_norm is not None:
                            # clip_grad_norm_ returns the total (pre-clip) L2 norm
                            # global norm clipping to preserve gradient direction (in contrast to value clipping)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=cfg.training.grad_clip_norm
                            ).item()
                        else:
                            # no clipping, but still compute the global L2 norm for logging
                            # torch version does not yet have torch.nn.utils.get_total_norm
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=float('inf')
                            ).item()
                        t1_4 = time.time()

                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.lr_scheduler.step()
                        t1_5 = time.time()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)
                        t1_6 = time.time()

                        # global_step counts number of optimizer updates
                        self.global_step += 1
                    else:
                        t1_5 = t1_6 = t1_4

                    # logging
                    train_losses.append(raw_loss.detach())
                    is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                    if (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                        if self.global_step % 50 == 0 or is_last_batch:
                            mean_loss = torch.stack(train_losses[-cfg.training.gradient_accumulate_every:]).mean().item()
                            
                            log_dict = {
                                'train_loss': mean_loss,
                                'total_l2_grad_norm': grad_norm,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': self.lr_scheduler.get_last_lr()[0],
                            }
                            
                            tepoch.set_postfix(loss=mean_loss, refresh=False)
                            wandb_run.log(log_dict, step=self.global_step)
                    t1_7 = time.time()

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1) \
                        and (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                        break

                    if verbose:
                        print(f" total one step time: {t1_7-t_before_next_batch:.3f}")
                        print(f" batch generation time {t1-t_before_next_batch:.3f}")
                        # True GPU H2D transfer time (CUDA events). At this point loss.backward()
                        # and the logging .item() have already synced the stream, so reading the
                        # event does not add a stall. The CPU-side numbers below (t1_1-t1 launch,
                        # t1_2-t1_1 augment) are kept for reference but are NOT the transfer cost.
                        if torch.cuda.is_available():
                            h2d_ms = h2d_start.elapsed_time(h2d_end)  # milliseconds
                            print(f" device transfer time (GPU, cuda-event): {h2d_ms/1000:.3f}")
                        print(f" device transfer launch (cpu): {t1_1-t1:.3f}")
                        print(f" augmentations time: {t1_2-t1_1:.3f}")
                        print(f" compute loss time: {t1_3-t1_2:.3f}")
                        print(f" diagnostic metrics time: {t1_4-t1_3:.3f}")
                        print(f" step optimizer time: {t1_5-t1_4:.3f}")
                        print(f" update ema time: {t1_6-t1_5:.3f}")
                        print(f" logging time: {t1_7-t1_6:.3f}")

                    t_before_next_batch = time.time()

            train_loss = torch.stack(train_losses).mean().detach()
            epoch_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()


            # run rollouts based on training and validation init poses
            if ((self.epoch + 1) % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                t3 = time.time()
                runner_log_train = env_runner.run(policy, dataset=dataset, prefix=f"train_epoch_{self.epoch}")
                runner_log_val = env_runner.run(policy, dataset=val_dataset, prefix=f"val_epoch_{self.epoch}")
                t4 = time.time()
                if verbose:
                    print(f"rollout time: {(t4-t3)/2:.3f}")
                
                # log rollouts with prefix
                for k, v in runner_log_train.items():
                    epoch_log[f"train_{k}"] = v
                for k, v in runner_log_val.items():
                    epoch_log[f"val_{k}"] = v

            
            # get validation loss
            if ((self.epoch + 1) % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            batch = self.val_data_augmentations(batch)
                            
                            loss, loss_dict = policy.compute_loss(batch)

                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.stack(val_losses)).detach()
                        # log epoch average validation loss
                        epoch_log['validation_loss'] = val_loss


            # run diffusion sampling on a training batch
            if ((self.epoch + 1) % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # use the fixed sampling batch (selected and augmented once before training)
                    obs_dict = train_sampling_batch['obs']
                    gt_action = train_sampling_batch['action']
                    
                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    epoch_log['train_action_mse_error'] = mse.detach()
                    del result
                    del pred_action
                    del mse
            
            
            if env_runner is None:
                # needed for checkpoint handling, TopKCheckpointManager looks at max scores 
                epoch_log['val_mean_success_rates'] = - train_loss

            policy.train()
            # ========= eval end for this epoch ==========
            
            # end of epoch — log epoch-level metrics (validation, rollout, etc.)
            epoch_log['global_step'] = self.global_step
            epoch_log['epoch'] = self.epoch
            wandb_run.log(epoch_log, step=self.global_step)

            # checkpoint
            if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in epoch_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)

            self.epoch += 1

    def eval(self, checkpoint_tag, use_dataset=False):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = self.get_checkpoint_path(tag=checkpoint_tag)
        if not lastest_ckpt_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint tag '{checkpoint_tag}' resolved to missing path: {lastest_ckpt_path}"
            )

        cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
        self.load_checkpoint(path=lastest_ckpt_path)
        
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        
        dataset = None
        if use_dataset:
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            assert isinstance(dataset, BaseDataset)
            val_dataset = dataset.get_validation_dataset()

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.to(torch.device(cfg.training.device))

        # compile so the diffusion sampling loop runs the accelerated submodules.
        # eval uses its own mode: an eval run is short relative to a 12-24h training run,
        # so a heavy max-autotune warmup usually will not amortize the way it does in run().
        if cfg.training.use_torch_compile:
            policy.apply_torch_compile(cfg.training.get('eval_torch_compile_mode', 'default'))

        if use_dataset:
            runner_log = env_runner.run(policy, prefix=f"seperate_eval_train_epoch_{self.epoch}", dataset=dataset)
            cprint(f"---------------- Eval Results - Train Dataset --------------", 'magenta')
            for key, value in runner_log.items():
                if isinstance(value, float):
                    cprint(f"{key}: {value:.4f}", 'magenta')

            runner_log = env_runner.run(policy, prefix=f"seperate_eval_val_epoch_{self.epoch}", dataset=val_dataset)
            cprint(f"---------------- Eval Results - Validation Dataset --------------", 'magenta')
            for key, value in runner_log.items():
                if isinstance(value, float):
                    cprint(f"{key}: {value:.4f}", 'magenta')
        
        runner_log = env_runner.run(policy, prefix=f"seperate_eval_test_epoch_{self.epoch}")
        cprint(f"---------------- Eval Results - Test Dataset --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-val_mean_success_rates={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('val_mean_success_rates=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            if tag.endswith('.ckpt'):
                return pathlib.Path(self.output_dir).joinpath('checkpoints', tag)
            else:
                return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
            
    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key in exclude_keys:
                continue
            if key not in self.__dict__:
                # e.g. lr_scheduler only exists during run(); eval-time loading
                # (eval(), create_from_checkpoint) has nothing to restore it into
                cprint(f"load_payload: skipping '{key}', workspace has no such attribute", 'yellow')
                continue
            self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
