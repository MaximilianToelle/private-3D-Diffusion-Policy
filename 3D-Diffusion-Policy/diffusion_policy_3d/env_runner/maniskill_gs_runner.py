
import torch
import numpy as np
import tqdm
import argparse
import time
import os
import imageio

import gsworld # Must be imported BEFORE arguments so it can securely inject into sys.path!
from gsworld.mani_skill.utils.wrappers import GSWorldWrapper
from arguments import PipelineParams
import gymnasium

from diffusion_policy_3d.env.maniskill.maniskill_gs_wrapper import GSManiskillDP3Wrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint


class GSManiSkillRunner(BaseRunner):
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
                 scene_gs_cfg_name="fr3_stack",
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        def env_fn(task_name):
            # Typically ManiSkill environments match task boundaries. Custom params should mirror run_with_gs.
            base_env = gymnasium.make(
                task_name, 
                robot_uids="fr3_umi",
                obs_mode="rgb+depth+segmentation",
                control_mode="pd_joint_pos",
                num_envs=n_envs,
                max_episode_steps=max_steps,
                sim_backend="gpu",
                sim_config=dict(sim_freq=100, control_freq=20),  # match data collection -> 5 physics substeps!
                )
            
            parser = argparse.ArgumentParser()
            robot_pipe = PipelineParams(parser)
            mapped_env = GSWorldWrapper(base_env, robot_pipe, scene_gs_cfg_name=scene_gs_cfg_name, device=device)
            
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    GSManiskillDP3Wrapper(mapped_env, num_gaussians=num_gaussians, use_gsplat_viewer=use_gsplat_viewer)
                ),
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

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval ManiSkill {self.task_name}", leave=False, mininterval=self.tqdm_interval_sec):
            
            init_state = None
            if dataset is not None:
                valid_episode_indices = np.where(dataset.global_train_mask)[0]
                random_episode_idx = np.random.choice(valid_episode_indices)
                init_state, _ = dataset.get_episode_init_data(random_episode_idx)
             
            obs = env.reset(options={'init_state': init_state} if init_state is not None else None)
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            while not done:
                obs_dict = dict_apply(dict(obs),
                                      lambda x: x.to(device=device) if isinstance(x, torch.Tensor)
                                      else torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['gs_positions'] = obs_dict['gs_positions'].unsqueeze(0)
                    obs_dict_input['gs_rotations_9d'] = obs_dict['gs_rotations_9d'].unsqueeze(0)
                    obs_dict_input['gs_log_scales'] = obs_dict['gs_log_scales'].unsqueeze(0)
                    obs_dict_input['gs_opacities'] = obs_dict['gs_opacities'].unsqueeze(0)
                    obs_dict_input['gs_rgb'] = obs_dict['gs_rgb'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                action_dict = dict_apply(action_dict,
                                          lambda x: x.detach())
                action = action_dict['action'].squeeze(0)

                #start = time.time()
                obs, reward, done, info = env.step(action)
                #print(f"Env step took: {time.time() - start}") 
                traj_reward += reward
                done = bool(done)
                
                # SAPIEN return metrics dict
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
            
            # The SimpleVideoRecordingWrapper records frames into memory.
            # We must explicitly extract them here before the next env.reset() clears them!
            try:
                video = env.env.get_video() # shape: (T, C, H, W)
                
                # 1. Save locally as an mp4 file for easy viewing
                video_dir = os.path.join(self.output_dir, "eval_videos")
                os.makedirs(video_dir, exist_ok=True)
                video_to_save = video.transpose(0, 2, 3, 1) # Convert to (T, H, W, C)
                out_path = os.path.join(video_dir, f"{prefix}_ep_{episode_idx}.mp4")
                imageio.mimsave(out_path, video_to_save, fps=self.fps, macro_block_size=1)
                cprint(f"Saved evaluation video to {out_path}", "cyan")
                
            except Exception as e:
                cprint(f"Failed to extract/save video from wrapper: {e}", "red")

        def _mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        log_data = dict()
        log_data['mean_traj_rewards'] = _mean(all_traj_rewards)
        log_data['mean_success_rates'] = _mean(all_success_rates)
        cprint(f"mean_success_rates: {_mean(all_success_rates)}", 'green')

        return log_data
