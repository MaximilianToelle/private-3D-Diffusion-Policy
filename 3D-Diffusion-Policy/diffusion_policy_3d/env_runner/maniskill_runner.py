"""Single, representation-agnostic ManiSkill evaluation runner.

  * The env's perception-specific wrappers are built by the `maniskill_env_obs_wrapper_builder`
    callable selected in config. Shared wrappers (video recording -> multi-step 
    -> optional relative-EE control) are applied in `__init__`. 
    The runner injects the shared runtime params (base_env, device, representation_space,
    n_obs_steps, n_action_steps); wrapper-specific params get defined in the config.
    -> For a new use case, add a new obs wrapper function + a config that points at it.
  * Logging / visualization live in `callbacks` (see `maniskill_callbacks.py`).

The eval loop forwards *every* observation key to the policy (the GSplat/DP3 encoders
select the keys they need and ignore the rest), plus any configured `obs_key_aliases`
(e.g. ``point_cloud: gs_positions``).
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import tqdm

import gsworld       # imported before gymnasium to register the GS wrappers' env side
import gsplat_envs   # registers our envs and agents (tasks moved out of the GSWorld fork)
import gymnasium
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.env_runner.maniskill_callbacks import EvalCallback
from diffusion_policy_3d.env.maniskill.relative_ee_control_wrapper import RelativeEEControlWrapper
from diffusion_policy_3d.env.maniskill.attention_overlay_wrapper import AttentionOverlayWrapper
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
        callbacks: Optional[List[EvalCallback]] = None,
        obs_key_aliases: Optional[Dict[str, str]] = None,
        eval_episodes: int = 20,
        tqdm_interval_sec: float = 5.0,
        task_name: Optional[str] = None,
        robot_uids: str = "fr3_umi_wrist435_modified",
        obs_mode: str = "rgb+depth+segmentation",
        representation_space: str = "relative_ee_pose",
        agent_proprio_dim: int = 11,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        n_envs: int = 1,
        max_steps: int = 1000,
        sim_freq: int = 100,
        control_freq: int = 20,    # match data collection -> 5 physics substeps!
        device: str = "cuda:0",
        realistic_depth: bool = False,
    ):
        super().__init__(output_dir)
        
        assert representation_space in REPRESENTATION_SPACES, \
            f"representation_space must be one of {REPRESENTATION_SPACES}, got '{representation_space}'"
        
        self.task_name = task_name
        self.representation_space = representation_space
        self.agent_proprio_dim = agent_proprio_dim      # policy-facing proprio dim from the config shape_meta
        self._obs_shape_validated = False
        self.eval_episodes = eval_episodes
        self.tqdm_interval_sec = tqdm_interval_sec
        self.device = device
        
        self.callbacks: List[EvalCallback] = list(callbacks) if callbacks is not None else []
        self.obs_key_aliases: Dict[str, str] = dict(obs_key_aliases) if obs_key_aliases is not None else {}

        if realistic_depth:
            assert n_envs == 1, "realistic depth only supports a single environment"
        base_env = gymnasium.make(
            task_name,
            robot_uids=robot_uids,
            obs_mode=obs_mode,
            control_mode="pd_ee_pos_quat" if representation_space in ("relative_ee_pose", "abs_ee_pose") else "pd_joint_pos",
            num_envs=n_envs,
            max_episode_steps=max_steps,
            sim_backend="gpu" if "cuda" in device else "cpu",
            sim_config=dict(sim_freq=sim_freq, control_freq=control_freq),
            realistic_depth=realistic_depth,
        )

        # `maniskill_env_obs_wrapper_builder` is a functools.partial from hydra with
        # perception-specific params defined in the config
        env = maniskill_env_obs_wrapper_builder(
            base_env=base_env,
            device=self.device,
        )

        env = AttentionOverlayWrapper(env)     # recolors the scene which gets captured in the recorded frames 
        env = SimpleVideoRecordingWrapper(env)
        env = MultiStepWrapper(
            env,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
            reward_agg_method='sum',
        )
        if representation_space == "relative_ee_pose":
            env = RelativeEEControlWrapper(env) # requires MultiStepWrapper underneath
        self.env = env

    def run(self, policy: BasePolicy, dataset=None, prefix: str = ""):
        policy_device = policy.device   # obs tensors must match the policy's weights, not the sim device
        env = self.env

        all_traj_rewards: List[float] = []
        all_success_rates: List[float] = []
        all_num_steps_to_success: List[float] = []

        for cb in self.callbacks:
            cb.at_run_start(self, policy)

        for episode_idx in tqdm.tqdm(range(self.eval_episodes),
                                     desc=f"Eval ManiSkill {self.task_name}",
                                     leave=False, mininterval=self.tqdm_interval_sec):

            init_state = None
            expert_trajectory = None
            if dataset is not None:
                # Pick a random episode from the global train/val mask.
                # NOTE: global_train_mask is set to the val_mask in the validation dataset.
                valid_episode_indices = np.where(dataset.global_train_mask)[0]
                assert len(valid_episode_indices) > 0, (
                    "the dataset holds no episodes to draw rollout initial states from -- "
                    "with val_ratio=0 and max_train_episodes unset nothing is held out for "
                    "validation, so pass dataset=None to roll out from random initial states")
                random_episode_idx = np.random.choice(valid_episode_indices)
                # get_episode_init_data handles multi-buffer routing internally.
                init_state, expert_trajectory = dataset.get_episode_init_data(random_episode_idx)

            obs = env.reset(options={'init_state': init_state} if init_state is not None else None)
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

            if is_success and isinstance(info, dict) and 'elapsed_steps' in info:
                all_num_steps_to_success.append(info["elapsed_steps"][-1].item())
            all_success_rates.append(float(is_success))
            all_traj_rewards.append(float(traj_reward))

            for cb in self.callbacks:
                self._safe_callback(
                    cb, 
                    "at_episode_end", 
                    episode_idx=episode_idx,
                    is_success=is_success, 
                    prefix=prefix, 
                    env=env,
                    expert_trajectory=expert_trajectory, 
                    info=info
                )

        def _mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        log_data = dict()
        log_data['mean_traj_rewards'] = _mean(all_traj_rewards)
        log_data['mean_success_rates'] = _mean(all_success_rates)
        cprint(f"mean_success_rates: {_mean(all_success_rates)}", 'green')

        if len(all_num_steps_to_success) > 0:
            log_data["steps_p25"] = np.percentile(all_num_steps_to_success, 25)
            log_data["steps_median"] = np.median(all_num_steps_to_success)
            log_data["steps_p75"] = np.percentile(all_num_steps_to_success, 75)
        else:
            # Penalize with max horizon to visually indicate failure
            log_data["steps_p25"] = env.max_episode_steps
            log_data["steps_median"] = env.max_episode_steps
            log_data["steps_p75"] = env.max_episode_steps

        for cb in self.callbacks:
            self._safe_callback(cb, "at_run_end", prefix=prefix, log_data=log_data)

        return log_data

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
            getattr(cb, method)(self, **kwargs)
        except Exception as e:
            cprint(f"[{cb.__class__.__name__}.{method}] failed: {e}", "red")


if __name__ == "__main__":
    import os
    import hydra
    import torch.nn as nn
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

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

    @hydra.main(
        version_base=None,
        config_path="../config",
        config_name="wrist_cam_dp3",
    )
    def main(cfg):
        output_dir = "test_eval_output"
        os.makedirs(output_dir, exist_ok=True)

        env_runner: BaseRunner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
        assert isinstance(env_runner, BaseRunner)

        action_dim = 8 if env_runner.representation_space == "abs_joint_pos" else 10
        policy = DummyPolicy(action_dim=action_dim, action_steps=8, device=env_runner.device)

        runner_log_train = env_runner.run(policy)
        print("Log data:", runner_log_train)

    main()
