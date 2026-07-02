
import torch
import numpy as np
import tqdm
import os
import imageio
import gymnasium

# DynaGSLAM config / params
from argparse import ArgumentParser
from utils.config_utils import read_config
from arguments import MapParams, OptimizationParams

from diffusion_policy_3d.env.maniskill.online_gsplat_wrapper import DynaGSLAMWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint


class DynaGSLAMManiSkillRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 tqdm_interval_sec=5.0,
                 n_envs=1,
                 task_name=None,
                 device="cuda:0",
                 num_gaussians=1024,
                 use_gsplat_viewer=False,
                 slam_config_path: str = None,   # path to a DynaGSLAM YAML config
                 cam_name: str = "right_cam",    # which ManiSkill camera to feed SLAM
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        # --- Load DynaGSLAM args from config file --------------------------------
        if slam_config_path is None:
            raise ValueError("slam_config_path must be provided for OnlineGSManiSkillRunner")
        slam_args = read_config(slam_config_path)

        # Enforce sim-compatible settings:
        #   • use_gt_pose=True  → skip ORB/ICP tracker, use sim camera extrinsics
        #   • mode='single process' → no multiprocessing overhead
        slam_args.use_gt_pose = True
        slam_args.mode = "single process"

        parser = ArgumentParser()
        _map_params = MapParams(parser)
        optimization_params = OptimizationParams(parser)
        optimization_params = optimization_params.extract(slam_args)
        # -------------------------------------------------------------------------

        def env_fn(task_name):
            base_env = gymnasium.make(
                task_name,
                robot_uids="fr3_umi",
                obs_mode="rgb+depth+segmentation",
                control_mode="pd_joint_pos",
                num_envs=n_envs,
                max_episode_steps=max_steps,
                sim_backend="gpu",
                sim_config=dict(sim_freq=100, control_freq=20),  # match data collection → 5 substeps
            )

            # Single wrapper replaces GSWorldWrapper + GSManiskillDP3Wrapper:
            #   • drives DynaGSLAM mapping from sim RGBD each step
            #   • returns the same policy-facing obs dict as the offline pipeline
            online_gs_env = DynaGSLAMWrapper(
                env=base_env,
                slam_args=slam_args,
                optimization_params=optimization_params,
                cam_name=cam_name,
                num_gaussians=num_gaussians,
                use_gsplat_viewer=use_gsplat_viewer,
            )

            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(online_gs_env),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)
        self.fps = fps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.logger_util_test3 = logger_util.LargestKRecorder(K=3)
        self.logger_util_test5 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy, dataset=None, prefix: str = ""):
        device = policy.device
        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        for episode_idx in tqdm.tqdm(
            range(self.eval_episodes),
            desc=f"Eval ManiSkill {self.task_name}",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        ):
            init_state = None
            if dataset is not None:
                replay_buffer = dataset.replay_buffer

                # The dataset uses a boolean mask to filter episodes for training/validation.
                # NOTE: Always using train_mask — it is set to val_mask in the validation dataset.
                valid_episode_indices = np.where(dataset.train_mask)[0]
                random_episode_idx = np.random.choice(valid_episode_indices)

                start_idx = (
                    replay_buffer.episode_ends[random_episode_idx - 1]
                    if random_episode_idx > 0
                    else 0
                )

                init_state = dict()
                if hasattr(dataset, 'actor_keys') and len(dataset.actor_keys) > 0:
                    init_state['actor_poses'] = {
                        k: replay_buffer[k][start_idx] for k in dataset.actor_keys
                    }
                init_state['agent_pos'] = replay_buffer['state'][start_idx]

            obs = env.reset(options={'init_state': init_state} if init_state is not None else None)
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False

            while not done:
                obs_dict = dict_apply(
                    dict(obs),
                    lambda x: x.to(device=device) if isinstance(x, torch.Tensor)
                    else torch.from_numpy(x).to(device=device),
                )

                with torch.no_grad():
                    obs_dict_input = {
                        'gs_positions':    obs_dict['gs_positions'].unsqueeze(0),
                        'gs_rotations_9d': obs_dict['gs_rotations_9d'].unsqueeze(0),
                        'gs_log_scales':   obs_dict['gs_log_scales'].unsqueeze(0),
                        'gs_opacities':    obs_dict['gs_opacities'].unsqueeze(0),
                        'gs_rgb':          obs_dict['gs_rgb'].unsqueeze(0),
                        'agent_pos':       obs_dict['agent_pos'].unsqueeze(0),
                    }
                    action_dict = policy.predict_action(obs_dict_input)

                action_dict = dict_apply(action_dict, lambda x: x.detach())
                action = action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)
                traj_reward += reward
                done = bool(done)

                # SAPIEN returns metrics in an info dict
                if isinstance(info, dict) and 'success' in info:
                    s = info['success']
                    if isinstance(s, torch.Tensor):
                        s = bool(s.any().item())
                    elif isinstance(s, np.ndarray):
                        s = bool(s.any())
                    else:
                        s = bool(s)
                    is_success = is_success or s

            all_success_rates.append(float(is_success))
            all_traj_rewards.append(float(traj_reward))

            # SimpleVideoRecordingWrapper records into memory; extract before next reset clears it.
            try:
                video = env.env.get_video()  # shape: (T, C, H, W)
                video_dir = os.path.join(self.output_dir, "eval_videos")
                os.makedirs(video_dir, exist_ok=True)
                video_to_save = video.transpose(0, 2, 3, 1)  # → (T, H, W, C)
                out_path = os.path.join(video_dir, f"{prefix}_ep_{episode_idx}.mp4")
                imageio.mimsave(out_path, video_to_save, fps=self.fps, macro_block_size=1)
                cprint(f"Saved evaluation video to {out_path}", "cyan")
            except Exception as e:
                cprint(f"Failed to extract/save video from wrapper: {e}", "red")

        def _mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        log_data = {
            'mean_traj_rewards':  _mean(all_traj_rewards),
            'mean_success_rates': _mean(all_success_rates),
        }
        cprint(f"mean_success_rates: {_mean(all_success_rates)}", 'green')
        return log_data