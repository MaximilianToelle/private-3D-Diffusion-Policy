"""Single, representation-agnostic ManiSkill evaluation runner.

  * A rollout from a dataset episode runs in the env that episode was recorded in: the
    recording's gymnasium.make arguments, stamped into the dataset by the converter, with only
    the evaluation's own ones (`evaluation_make_kwargs`) in their place. `_env` builds one env
    per recorded setup, e.g. per piece.
  * The env's perception-specific wrappers are built by the `maniskill_env_obs_wrapper_builder`
    callable selected in config. Shared wrappers (video recording -> actor pose recording -> multi-step 
    -> optional relative-EE control) are applied in `_env`.
    The runner injects the shared runtime params (base_env, device, representation_space,
    n_obs_steps, n_action_steps); wrapper-specific params get defined in the config.
    -> For a new use case, add a new obs wrapper function + a config that points at it.
  * With num_workers > 1, worker processes roll out the episodes in parallel, each with its own
    envs, callbacks and copy of the policy (`_rollout_worker`).
  * Logging / visualization live in `callbacks` (see `maniskill_callbacks.py`).

The eval loop forwards *every* observation key to the policy (the GSplat/DP3 encoders
select the keys they need and ignore the rest), plus any configured `obs_key_aliases`
(e.g. ``point_cloud: gs_positions``).
"""

import copy
import json
import pickle
import traceback
from multiprocessing.connection import wait
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.multiprocessing
import tqdm

import gsplat_envs   # registers our envs and agents (tasks moved out of the GSWorld fork)
import gymnasium
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.dataset.multi_memmap_dataset import discover_dataset_paths, recorded_env
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.env_runner.maniskill_callbacks import EvalCallback
from diffusion_policy_3d.env.maniskill.relative_ee_control_wrapper import RelativeEEControlWrapper
from diffusion_policy_3d.env.maniskill.attention_overlay_wrapper import AttentionOverlayWrapper
from diffusion_policy_3d.env.maniskill.actor_pose_recording_wrapper import ActorPoseRecordingWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from termcolor import cprint


# Canonical set of representations the pipeline supports. The runner is the
# single entry point where `representation_space` fans out (control_mode, the RelativeEE
# wrapper, and the obs-wrapper builder), so it validates the value once, up front.
# "abs_ee_pose": absolute pd_ee_pos_quat targets straight from the policy (mindmap
# baseline predicts absolute keyposes) -- no RelativeEEControlWrapper.
REPRESENTATION_SPACES = ("relative_ee_pose", "abs_joint_pos", "abs_ee_pose")


def _extract_success(info) -> bool:
    """SAPIEN returns a metrics dict; turn its `success` field to a python bool."""
    if isinstance(info, dict) and 'success' in info:
        s = info['success']
        if isinstance(s, torch.Tensor):
            return bool(s.any().item())
        if isinstance(s, np.ndarray):
            return bool(s.any())
        return bool(s)
    return False


class ManiSkillRunner(BaseRunner):
    def __init__(
        self,
        output_dir,
        maniskill_env_obs_wrapper_builder: Callable,
        dataset_path,               # the policy's datasets; a rollout without a dataset runs in the env of the first
        callbacks: Optional[List[EvalCallback]] = None,                       # the shared ones (base.yaml)
        scene_representation_callbacks: Optional[List[EvalCallback]] = None,  # the scene representation's
        task_callbacks: Optional[List[EvalCallback]] = None,                  # the task's
        obs_key_aliases: Optional[Dict[str, str]] = None,
        eval_episodes: int = 20,
        num_workers: int = 1,       # processes rolling out the episodes in parallel
        tqdm_interval_sec: float = 5.0,
        task_name: Optional[str] = None,    # the env id, asserted against the datasets' stamp
        obs_mode: str = "rgb+depth+segmentation",
        representation_space: str = "relative_ee_pose",
        agent_proprio_dim: int = 11,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        max_steps: int = 1000,
        device: str = "cuda:0",
        depth_mode: str = None,     # None (gt) | fast_foundation_stereo, the depth the datasets were converted from
    ):
        # a rollout worker builds its own runner from these
        self.init_kwargs = {key: value for key, value in locals().items() if key not in ("self", "__class__")}
        super().__init__(output_dir)
        
        assert representation_space in REPRESENTATION_SPACES, \
            f"representation_space must be one of {REPRESENTATION_SPACES}, got '{representation_space}'"
        
        self.task_name = task_name
        self.representation_space = representation_space
        self.agent_proprio_dim = agent_proprio_dim      # policy-facing proprio dim from the config shape_meta
        self._obs_shape_validated = False
        self.eval_episodes = eval_episodes
        self.num_workers = num_workers
        self.tqdm_interval_sec = tqdm_interval_sec
        self.device = device
        
        # each owner contributes its own callbacks; their hooks run in this order
        self.callbacks: List[EvalCallback] = [
            *(scene_representation_callbacks or []), *(callbacks or []), *(task_callbacks or [])]
        self.obs_key_aliases: Dict[str, str] = dict(obs_key_aliases) if obs_key_aliases is not None else {}

        # The gymnasium.make arguments the evaluation sets itself; all others are the recording's.
        # None of them changes the scene, the physics or the cameras.
        self.evaluation_make_kwargs = dict(
            obs_mode=obs_mode,              # which images ManiSkill renders in its step
            depth_mode=depth_mode,          # which depth the cameras add (the builders assert the dataset's)
            # the controller of the policy's actions, the recordings' pd_joint_pos for abs_joint_pos
            control_mode="pd_ee_pos_quat" if representation_space in ("relative_ee_pose", "abs_ee_pose") else "pd_joint_pos",
            max_episode_steps=max_steps,    # when a rollout that has not succeeded stops
        )
        # `maniskill_env_obs_wrapper_builder` is a functools.partial from hydra with
        # perception-specific params defined in the config
        self.maniskill_env_obs_wrapper_builder = maniskill_env_obs_wrapper_builder
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps

        self.envs = {}
        self.env = self._env(recorded_env(discover_dataset_paths(dataset_path)[0]))
        self.workers = []   # connections to the rollout worker processes, started on the first run

    def _env(self, recorded: dict):
        """The wrapped env a demonstration was recorded in (recorded_env of its dataset), with the
        evaluation's own make arguments in place of the recorded ones; built on first use."""
        env_id = recorded['env_id']
        assert env_id == self.task_name, f"the dataset was recorded in {env_id}, but the task's env is {self.task_name}"
        # The env sets its own physics and control rate, so the recorded rate is checked, not passed.
        make_kwargs = {key: value for key, value in recorded['env_kwargs'].items() if key != "sim_config"}
        make_kwargs.update(self.evaluation_make_kwargs)
        key = json.dumps([env_id, make_kwargs], sort_keys=True)
        if key not in self.envs:
            base_env = gymnasium.make(env_id, **make_kwargs)
            rate = (base_env.unwrapped.sim_freq, base_env.unwrapped.control_freq)
            assert rate == (recorded['sim_freq'], recorded['control_freq']), (
                f"{env_id} runs at sim_freq/control_freq {rate}, its demonstrations were recorded at "
                f"{(recorded['sim_freq'], recorded['control_freq'])}")
            env = self.maniskill_env_obs_wrapper_builder(base_env=base_env, device=self.device)
            env = AttentionOverlayWrapper(env)     # recolors the scene which gets captured in the recorded frames
            env = SimpleVideoRecordingWrapper(env)
            env = ActorPoseRecordingWrapper(env, base_env.unwrapped)
            env = MultiStepWrapper(
                env,
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.n_action_steps,
                max_episode_steps=self.max_steps,
                reward_agg_method='sum',
            )
            if self.representation_space == "relative_ee_pose":
                env = RelativeEEControlWrapper(env) # requires MultiStepWrapper underneath
            self.envs[key] = env
        return self.envs[key]

    def run(self, policy: BasePolicy, dataset=None, prefix: str = "", episode_seed: Optional[int] = None):
        """Rolls out eval_episodes episodes, from the start states of the dataset's episodes if
        given: those of its global train/val mask, shuffled with episode_seed (in order if None),
        cycling when eval_episodes exceeds them. train.py passes the epoch, so each evaluation
        draws other episodes, and runs of the same experiment draw the same ones at an epoch.
        With num_workers > 1 worker processes roll them out in parallel. Every rollout starts in a
        rebuilt scene and seeds the policy's sampling noise with episode_seed and its index, so it
        does not depend on the process it runs in or on the episodes before it, every evaluation
        draws new noise, and runs evaluated at the same epoch draw the same."""
        self.rollout_dataset = dataset   # callbacks read the episodes and their split from it
        episodes = self._episodes(dataset, episode_seed)
        if self.num_workers > 1:
            results = self._run_in_workers(policy, episodes, prefix)
        else:
            for cb in self.callbacks:
                cb.at_run_start(self, policy)
            results = [self._run_episode(policy, episode, prefix)
                       for episode in tqdm.tqdm(episodes, desc=f"Eval ManiSkill {self.task_name}",
                                                leave=False, mininterval=self.tqdm_interval_sec)]

        success_rates = [float(result['is_success']) for result in results]
        steps_to_success = [result['steps_to_success'] for result in results if result['steps_to_success'] is not None]
        log_data = dict()
        log_data['mean_traj_rewards'] = np.mean([result['traj_reward'] for result in results])
        log_data['mean_success_rates'] = np.mean(success_rates)
        cprint(f"mean_success_rates: {log_data['mean_success_rates']}", 'green')

        if len(steps_to_success) > 0:
            log_data["steps_p25"] = np.percentile(steps_to_success, 25)
            log_data["steps_median"] = np.median(steps_to_success)
            log_data["steps_p75"] = np.percentile(steps_to_success, 75)
        else:
            # Penalize with max horizon to visually indicate failure
            log_data["steps_p25"] = self.max_steps
            log_data["steps_median"] = self.max_steps
            log_data["steps_p75"] = self.max_steps

        for index, cb in enumerate(self.callbacks):
            self._safe_callback(cb, "at_run_end", prefix=prefix, log_data=log_data,
                                records=[result['callback_records'][index] for result in results])

        return log_data

    def _episodes(self, dataset, episode_seed):
        """What each rollout needs, in rollout order: its index, the seed of the policy's sampling
        noise, its start state, its demonstration and the env it was recorded in (None without a
        dataset: random start states in the env of the first dataset)."""
        noise_seeds = [index if episode_seed is None else episode_seed * self.eval_episodes + index
                       for index in range(self.eval_episodes)]
        if dataset is None:
            return [dict(index=index, noise_seed=noise_seeds[index], init_state=None, expert_trajectory=None,
                         recorded_env=None) for index in range(self.eval_episodes)]
        # NOTE: global_train_mask is set to the val_mask in the validation dataset.
        dataset_episodes = np.where(dataset.global_train_mask)[0]
        assert len(dataset_episodes) > 0, (
            "the dataset holds no episodes to draw rollout initial states from -- "
            "with val_ratio=0 and max_train_episodes unset nothing is held out for "
            "validation, so pass dataset=None to roll out from random initial states")
        if episode_seed is not None:
            dataset_episodes = np.random.default_rng(episode_seed).permutation(dataset_episodes)
        episodes = []
        for index in range(self.eval_episodes):
            dataset_episode = dataset_episodes[index % len(dataset_episodes)]
            # get_episode_init_data handles multi-buffer routing internally.
            init_state, expert_trajectory = dataset.get_episode_init_data(dataset_episode)
            episodes.append(dict(index=index, noise_seed=noise_seeds[index], init_state=init_state,
                                 expert_trajectory=expert_trajectory,
                                 recorded_env=dataset.episode_recorded_env(dataset_episode)))
        return episodes

    def _run_episode(self, policy: BasePolicy, episode: dict, prefix: str) -> dict:
        """One rollout (an entry of _episodes): whether it succeeded, its reward, its steps to
        success and every callback's record of it."""
        policy_device = policy.device   # obs tensors must match the policy's weights, not the sim device
        episode_idx, init_state, expert_trajectory = episode['index'], episode['init_state'], episode['expert_trajectory']
        env = self.env if episode['recorded_env'] is None else self._env(episode['recorded_env'])

        with torch.random.fork_rng(devices=[policy_device]):
            torch.manual_seed(episode['noise_seed'])
            # A rebuilt scene (reconfigure, 0.07 s), since PhysX results depend on the scene's
            # earlier episodes, which differ between the worker processes.
            options = dict(reconfigure=True) if init_state is None else dict(reconfigure=True, init_state=init_state)
            obs = env.reset(options=options)
            if not self._obs_shape_validated:
                self._validate_obs_shape(obs)
            policy.reset()

            for cb in self.callbacks:
                cb.before_episode_start_after_env_reset(self, episode_idx=episode_idx,
                                    init_state=init_state, expert_trajectory=expert_trajectory)

            done = False
            traj_reward = 0
            is_success = False
            info = None

            while not done:
                obs_dict = dict_apply(dict(obs),
                                      lambda x: x.to(device=policy_device) if isinstance(x, torch.Tensor)
                                      else torch.from_numpy(x).to(device=policy_device))

                with torch.no_grad():
                    # Forward every obs key (encoders select what they need); add batch dim and alias keys
                    obs_dict_input = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
                    for alias, src in self.obs_key_aliases.items():
                        obs_dict_input[alias] = obs_dict[src].unsqueeze(0)
                        
                    action_dict = policy.predict_action(obs_dict_input)

                action_dict = dict_apply(action_dict, lambda x: x.detach())
                action = action_dict['action'].squeeze(0)

                for cb in self.callbacks:
                    cb.after_action_predict(
                        self, 
                        obs_dict=obs_dict, 
                        action=action,
                        action_dict=action_dict, 
                        policy=policy, 
                        env=env
                    )

                obs, reward, done, info = env.step(action)

                for cb in self.callbacks:
                    cb.after_env_step(
                        self, 
                        obs=obs, 
                        action=action, 
                        reward=reward,
                        done=done, 
                        info=info, 
                        policy=policy, 
                        env=env
                    )

                traj_reward += reward
                done = bool(done)
                is_success = is_success or _extract_success(info)

        steps_to_success = None
        if is_success and isinstance(info, dict) and 'elapsed_steps' in info:
            steps_to_success = info["elapsed_steps"][-1].item()
        callback_records = [
            self._safe_callback(cb, "at_episode_end", episode_idx=episode_idx, is_success=is_success, prefix=prefix,
                                env=env, expert_trajectory=expert_trajectory, info=info)
            for cb in self.callbacks]
        return dict(index=episode_idx, is_success=is_success, traj_reward=float(traj_reward),
                    steps_to_success=steps_to_success, callback_records=callback_records)

    def _run_in_workers(self, policy: BasePolicy, episodes: List[dict], prefix: str) -> List[dict]:
        """_run_episode of every episode, in num_workers worker processes, each taking the next
        episode as soon as it has finished one; the results in episode order."""
        # The policy and its weights go to the workers as pickled bytes. Tensors sent as they are
        # travel as shared-memory file descriptors, of which a cluster process may open 1024.
        if not self.workers:
            context = torch.multiprocessing.get_context("spawn")
            worker_policy = pickle.dumps(copy.deepcopy(policy).cpu())
            worker_kwargs = dict(self.init_kwargs, num_workers=1)
            for _ in range(self.num_workers):
                connection, worker_connection = context.Pipe()
                context.Process(target=_rollout_worker, args=(worker_kwargs, worker_policy, worker_connection),
                                daemon=True).start()
                self.workers.append(connection)

        state_dict = pickle.dumps({key: value.detach().cpu() for key, value in policy.state_dict().items()})
        for connection in self.workers:
            connection.send(('run', state_dict, prefix))
        pending = list(episodes)
        busy = []
        for connection in self.workers[:len(pending)]:
            connection.send(('episode', pending.pop(0)))
            busy.append(connection)
        results = {}
        with tqdm.tqdm(total=len(episodes), desc=f"Eval ManiSkill {self.task_name} ({self.num_workers} workers)",
                       leave=False, mininterval=self.tqdm_interval_sec) as progress:
            while busy:
                for connection in wait(busy):
                    status, payload = connection.recv()
                    if status == 'error':
                        raise RuntimeError(f"a rollout worker failed:\n{payload}")
                    results[payload['index']] = payload
                    progress.update()
                    if pending:
                        connection.send(('episode', pending.pop(0)))
                    else:
                        busy.remove(connection)
        return [results[episode['index']] for episode in episodes]

    def _validate_obs_shape(self, obs):
        """
        Fail fast if the FULL wrapper stack's proprio dim disagrees with the config
        shape_meta. Runs once, on the first reset.
        """
        actual = obs['agent_proprio'].shape[-1]
        assert actual == self.agent_proprio_dim, (
            f"agent_proprio dim mismatch: config shape_meta declares {self.agent_proprio_dim} "
            f"but the full env stack produced {actual} for "
            f"representation_space='{self.representation_space}'."
        )
        self._obs_shape_validated = True

    def _safe_callback(self, cb: EvalCallback, method: str, **kwargs):
        try:
            return getattr(cb, method)(self, **kwargs)
        except Exception as e:
            cprint(f"[{cb.__class__.__name__}.{method}] failed: {e}", "red")


def _rollout_worker(runner_kwargs: dict, pickled_policy: bytes, connection):
    """A rollout worker process: builds its own runner (envs and callbacks) and rolls out the
    episodes the main process sends, with the weights it sends at the start of every run."""
    torch.set_num_threads(1)
    runner = ManiSkillRunner(**runner_kwargs)
    policy = pickle.loads(pickled_policy).eval()
    while True:
        try:
            message = connection.recv()
        except EOFError:    # the main process has closed the connection or ended
            return
        try:
            if message[0] == 'run':
                _, pickled_state_dict, prefix = message
                # after loading, since the normalizer loads its state as new tensors on the state's device
                policy.load_state_dict(pickle.loads(pickled_state_dict))
                policy.to(runner.device)
                for cb in runner.callbacks:
                    cb.at_run_start(runner, policy)
            else:
                connection.send(('result', runner._run_episode(policy, message[1], prefix)))
        except Exception:
            connection.send(('error', traceback.format_exc()))


if __name__ == "__main__":
    import os
    import hydra
    import torch.nn as nn
    from omegaconf import OmegaConf

    from diffusion_policy_3d.common.config_resolvers import register_config_resolvers

    register_config_resolvers()

    class DummyExtractor:
        def __init__(self):
            self._latest_pool_indices = torch.zeros(1024, dtype=torch.long)

    class DummyEncoder:
        def __init__(self):
            self.extractor = DummyExtractor()

    class DummyPolicy(nn.Module):
        def __init__(self, action_dim=7, action_steps=8, device="cuda:0"):
            super().__init__()
            self._device = torch.device(device)
            self.action_dim = action_dim
            self.action_steps = action_steps
            self.obs_encoder = DummyEncoder()

        @property
        def device(self):
            return self._device

        def reset(self):
            pass

        def eval(self):
            pass

        def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            self.obs_encoder.extractor._latest_pool_indices = torch.arange(1024, dtype=torch.long, device=self.device)
            action = torch.rand((1, self.action_steps, self.action_dim), device=self.device)
            return {'action': action}

    # Smoke test of the runner without a trained checkpoint.
    #   default: random DummyPolicy on the config's task
    #     python -m diffusion_policy_3d.env_runner.maniskill_runner
    #   Fawad's motion-planning oracle (gsplat_envs) on the DynaGSLAM task; it emits joint targets,
    #   so the task has to run in joint space:
    #     python -m diffusion_policy_3d.env_runner.maniskill_runner --config-name experiment/stack_dynagslam \
    #         policy.representation_space=abs_joint_pos +test_policy=motion_planning
    @hydra.main(
        version_base=None,
        config_path="../config",
        config_name="experiment/stack_dynagslam",
    )
    def main(cfg):
        output_dir = "test_eval_output"
        os.makedirs(output_dir, exist_ok=True)

        env_runner: BaseRunner = hydra.utils.instantiate(cfg.env_runner, output_dir=output_dir)
        assert isinstance(env_runner, BaseRunner)

        test_policy = cfg.get("test_policy", "dummy")
        if test_policy == "dummy":
            action_dim = 8 if env_runner.representation_space == "abs_joint_pos" else 10
            policy = DummyPolicy(action_dim=action_dim, action_steps=8, device=env_runner.device)
        elif test_policy == "motion_planning":
            from gsplat_envs.mani_skill.motion_planning.franka.motionplanner_policy import CanStackMotionPlanningPolicy
            assert env_runner.representation_space == "abs_joint_pos", \
                "CanStackMotionPlanningPolicy emits joint targets: run with policy.representation_space=abs_joint_pos"
            # The obs wrapper is a gym.Env, so .unwrapped stops there; .env below it is the ManiSkill env.
            policy = CanStackMotionPlanningPolicy(
                env=env_runner.env.unwrapped.env,
                action_steps=cfg.n_action_steps,
                device=env_runner.device,
            )
        else:
            raise ValueError(f"unknown test_policy '{test_policy}', expected 'dummy' or 'motion_planning'")

        runner_log_train = env_runner.run(policy)
        print("Log data:", runner_log_train)

    main()
