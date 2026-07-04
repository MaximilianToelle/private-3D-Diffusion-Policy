
import torch
import torch.nn as nn
import numpy as np
import tqdm
import argparse
import time
import os
from typing import Dict
import imageio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import cv2

import gsworld # Must be imported BEFORE arguments so it can securely inject into sys.path!
from gsworld.mani_skill.utils.wrappers import WristCamGSWorldWrapper
from arguments import PipelineParams
import gymnasium

from diffusion_policy_3d.env.maniskill.maniskill_wrist_cam_gs_wrapper import WristCamGSManiskillDP3Wrapper
from diffusion_policy_3d.env.maniskill.relative_ee_control_wrapper import RelativeEEControlWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint


class WristCamGSManiSkillRunner(BaseRunner):
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
                 min_opacity=0.9,
                 use_gsplat_viewer=False,
                 scene_gs_cfg_name="fr3_stack",
                 representation_space="relative_ee_pose",
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        assert representation_space in ("abs_joint_pos", "relative_ee_pose"), \
            f"representation_space must be 'abs_joint_pos' or 'relative_ee_pose', got '{representation_space}'"
        self.representation_space = representation_space

        def env_fn(task_name):
            # Typically ManiSkill environments match task boundaries. Custom params should mirror run_with_gs.
            base_env = gymnasium.make(
                task_name, 
                robot_uids="fr3_umi_wrist435_modified",
                obs_mode="rgb+depth+segmentation",
                control_mode="pd_ee_pose" if representation_space == "relative_ee_pose" else "pd_joint_pos",
                num_envs=n_envs,
                max_episode_steps=max_steps,
                sim_backend="gpu",
                sim_config=dict(sim_freq=100, control_freq=20),  # match data collection -> 5 physics substeps!
                )
            
            parser = argparse.ArgumentParser()
            robot_pipe = PipelineParams(parser)
            mapped_env = WristCamGSWorldWrapper(
                base_env, 
                robot_pipe, 
                scene_gs_cfg_name=scene_gs_cfg_name, 
                device=device, 
                use_gsplat_viewer=use_gsplat_viewer,
            )
            
            wrapped_env = MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    WristCamGSManiskillDP3Wrapper(
                        mapped_env, 
                        representation_space,
                        num_gaussians=num_gaussians,
                        min_opacity=min_opacity,
                        n_action_steps=n_action_steps,
                        n_obs_steps=n_obs_steps,
                    )
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
            
            if representation_space == "relative_ee_pose":
                wrapped_env = RelativeEEControlWrapper(wrapped_env)
                
            return wrapped_env

        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)
        self.fps = fps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BasePolicy, dataset=None, prefix: str = ""):
        device = policy.device
        all_traj_rewards = []
        all_success_rates = []
        all_num_steps_to_success = []
        
        env = self.env

        all_expert_trajectories = []
        all_policy_trajectories = []
        all_predicted_actions = []
        all_num_unique_gaussians = []

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval ManiSkill {self.task_name}", leave=False, mininterval=self.tqdm_interval_sec):
            
            init_state = None
            expert_q_sequence = None
            
            if dataset is not None:
                # Pick a random episode from the global train/val mask
                # NOTE: global_train_mask is set to the val_mask in the validation dataset
                valid_episode_indices = np.where(dataset.global_train_mask)[0]
                random_episode_idx = np.random.choice(valid_episode_indices)
                
                # get_episode_init_data handles multi-buffer routing internally
                init_state, expert_q_sequence = dataset.get_episode_init_data(random_episode_idx)

            obs = env.reset(options={'init_state': init_state} if init_state is not None else None)
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False

            current_predicted_actions = []
            current_num_unique_gaussians = []

            while not done:
                obs_dict = dict_apply(dict(obs),
                                      lambda x: x.to(device=device) if isinstance(x, torch.Tensor)
                                      else torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['gs_positions'] = obs_dict['gs_positions'].unsqueeze(0)
                    obs_dict_input['point_cloud'] = obs_dict['gs_positions'].unsqueeze(0)
                    obs_dict_input['gs_rotations_9d'] = obs_dict['gs_rotations_9d'].unsqueeze(0)
                    obs_dict_input['gs_surface_normals'] = obs_dict['gs_surface_normals'].unsqueeze(0)
                    obs_dict_input['gs_log_scales'] = obs_dict['gs_log_scales'].unsqueeze(0)
                    obs_dict_input['gs_opacities'] = obs_dict['gs_opacities'].unsqueeze(0)
                    obs_dict_input['gs_rgb'] = obs_dict['gs_rgb'].unsqueeze(0)
                    obs_dict_input['agent_proprio'] = obs_dict['agent_proprio'].unsqueeze(0)
                    
                    action_dict = policy.predict_action(obs_dict_input)

                pool_idx = policy.obs_encoder.extractor._latest_pool_indices
                if len(pool_idx.shape) > 1:
                    pool_idx = pool_idx[-1]
                    
                current_num_unique_gaussians.append(len(torch.unique(pool_idx)))
                
                if len(obs_dict['gs_rgb'].shape) > 1:
                    obs_gs_rgb = obs_dict['gs_rgb'][-1]
                # Inject the selected Gaussians back into the full original Gaussian scene
                apply_heatmap_to_full_scene(
                    env_wrapper=env.unwrapped, 
                    pool_idx=pool_idx, 
                    obs_gs_rgb=obs_gs_rgb
                )

                action_dict = dict_apply(action_dict,
                                          lambda x: x.detach())
                action = action_dict['action'].squeeze(0)

                # Log predicted action targets (what the policy wants the PD controller to reach)
                current_predicted_actions.extend(action.cpu().numpy())

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

            all_expert_trajectories.append(expert_q_sequence)
            all_policy_trajectories.append(torch.stack(list(env.traj_agent_proprio)).cpu().numpy())
            all_predicted_actions.append(np.array(current_predicted_actions))
            all_num_unique_gaussians.append(np.array(current_num_unique_gaussians))

            if is_success:
                all_num_steps_to_success.append(info["elapsed_steps"][-1].item())

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
                video_to_save = add_legend_to_video(video_to_save, colormap='cool')
                if is_success:
                    out_path = os.path.join(video_dir, f"{prefix}_ep_{episode_idx}_success.mp4")
                else:
                    out_path = os.path.join(video_dir, f"{prefix}_ep_{episode_idx}_failure.mp4")
                imageio.mimsave(out_path, video_to_save, fps=self.fps, macro_block_size=1)
                cprint(f"Saved evaluation video to {out_path}", "cyan")
                
            except Exception as e:
                cprint(f"Failed to extract/save video from wrapper: {e}", "red")

        try:
            plot_dir = os.path.join(self.output_dir, "eval_plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"{prefix}_phase_corridor.png")
            joint_pos_path = os.path.join(plot_dir, f"{prefix}_joint_pos_time.png")
            num_unique_gs_path = os.path.join(plot_dir, f"{prefix}_num_unique_gaussians_time.png")
            
            control_dt = 1.0 / self.env.unwrapped.env.unwrapped.control_freq 
            
            save_phase_corridor_plot(
                all_expert_trajectories, 
                all_policy_trajectories, 
                dt=control_dt, 
                save_path=plot_path
            )
            cprint(f"Saved Phase Corridor plot to {plot_path}", "cyan")
            
            save_joint_pos_over_time_plot(
                all_expert_trajectories, 
                all_policy_trajectories,
                all_predicted_actions,
                save_path=joint_pos_path
            )
            cprint(f"Saved Joint Position plot to {joint_pos_path}", "cyan")
            
            save_num_unique_gaussians_plot(
                all_num_unique_gaussians,
                save_path=num_unique_gs_path
            )
            cprint(f"Saved Num Unique Gaussians plot to {num_unique_gs_path}", "cyan")
            
        except Exception as e:
            cprint(f"Failed to generate plots: {e}", "red")


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
        
        return log_data


def save_phase_corridor_plot(
    expert_trajectories: list, 
    policy_trajectories: list, 
    dt: float, 
    save_path: str
):
    """ Generates the Velocity vs Position Phase Plot for all joints."""
    
    if len(policy_trajectories) == 0:
        return None

    # Get num joints from the first trajectory
    if len(expert_trajectories) > 0 and expert_trajectories[0] is not None:
        num_joints = expert_trajectories[0].shape[1]
    else:
        num_joints = policy_trajectories[0].shape[1]
    
    fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(10, 3 * num_joints))
    if num_joints == 1: axes = [axes]
    fig.suptitle("Phase Corridor: Policy vs. Ground Truth", fontsize=16, fontweight='bold')
    
    for j in range(num_joints):
        ax = axes[j]
        
        # Plot all Policy Rollouts (Thin, translucent red)
        for i, p_q in enumerate(policy_trajectories):
            p_dq = np.gradient(p_q, dt, axis=0)
            p_dq[0] = 0.0   # np.gradient is inaccurate for t=0. Maniskill resets agent to zero velocity! 
            label = 'Policy Rollout' if i == 0 else "_nolegend_"
            ax.plot(p_q[:, j], p_dq[:, j], color='red', alpha=0.3, linewidth=1.0, label=label)

        # Plot all Expert Ground Truths (Thick, black, dashed)
        for i, e_q in enumerate(expert_trajectories):
            if e_q is not None:
                e_dq = np.gradient(e_q, dt, axis=0)
                e_dq[0] = 0.0   # np.gradient is inaccurate for t=0. Maniskill resets agent to zero velocity! 
                label = 'Motion Planner (Expert)' if i == 0 else "_nolegend_"
                ax.plot(e_q[:, j], e_dq[:, j], color='black', linewidth=2, label=label)
                # Mark start and end targets
                ax.scatter(e_q[0, j], e_dq[0, j], c='green', marker='o', s=100, zorder=5, label='Start' if i == 0 else "_nolegend_")
                ax.scatter(e_q[-1, j], e_dq[-1, j], c='black', marker='*', s=150, zorder=5, label='End Target' if i == 0 else "_nolegend_")

        ax.set_title(f"Joint {j}")
        ax.set_ylabel("Velocity [rad/s]")
        ax.grid(True, alpha=0.3)
        if j == 0: ax.legend(loc='upper right')
        if j == num_joints - 1: ax.set_xlabel("Position [rad]")

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def save_joint_pos_over_time_plot(
    expert_trajectories: list, 
    policy_trajectories: list,
    predicted_actions: list = None,
    save_path: str = "joint_pos_time.png"
):
    """ Generates the Position vs Timestep Plot for all joints.
    
    Shows three lines per joint:
    - Ground Truth (expert): black solid line
    - Predicted Actions (policy targets): blue dashed — what the policy wants
    - Actual qpos (after PD control): red solid — what actually happened
    
    The gap between predicted and actual reveals PD execution error.
    The gap between ground truth and predicted reveals policy prediction error.
    """
    if len(policy_trajectories) == 0:
        return None

    # Get num joints from the first trajectory
    if len(expert_trajectories) > 0 and expert_trajectories[0] is not None:
        num_joints = expert_trajectories[0].shape[1]
    else:
        num_joints = policy_trajectories[0].shape[1]
    
    fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(12, 3 * num_joints))
    if num_joints == 1: axes = [axes]
    fig.suptitle("Joint Position over Time: Predicted vs. Actual vs. Ground Truth", fontsize=16, fontweight='bold')
    
    for j in range(num_joints):
        ax = axes[j]
        
        # Plot all Actual qpos
        for i, p_q in enumerate(policy_trajectories):
            time_axis = np.arange(len(p_q))
            label = 'Actual qpos (after PD)' if i == 0 else "_nolegend_"
            ax.plot(time_axis, p_q[:, j], color='red', alpha=0.3, linewidth=1.0, label=label)

        # Plot all Expert Ground Truths (Thick, black)
        for i, e_q in enumerate(expert_trajectories):
            if e_q is not None:
                time_axis = np.arange(len(e_q))
                label = 'Motion Planner (Expert)' if i == 0 else "_nolegend_"
                ax.plot(time_axis, e_q[:, j], color='black', linewidth=2, label=label)

        # Plot all Predicted Actions
        for i, p_a in enumerate(predicted_actions):
            if p_a is not None and j < p_a.shape[1]:
                time_axis = np.arange(len(p_a))
                label = 'Predicted Action (PD target)' if i == 0 else "_nolegend_"
                ax.plot(time_axis, p_a[:, j], color='dodgerblue', alpha=0.4, linewidth=1.0, linestyle='--', label=label)

        ax.set_title(f"Joint {j}")
        ax.set_ylabel("Position [rad]")
        ax.grid(True, alpha=0.3)
        if j == 0: ax.legend(loc='upper right')
        if j == num_joints - 1: ax.set_xlabel("Timestep")

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def save_num_unique_gaussians_plot(
    all_num_unique_gaussians: list,
    save_path: str
):
    """ Plots median, 25%, and 75% quantile of num unique Gaussians over time. """
    if len(all_num_unique_gaussians) == 0:
        return None
        
    # Find max length
    max_len = max(len(traj) for traj in all_num_unique_gaussians)
    
    # Pad with NaN
    padded_data = np.full((len(all_num_unique_gaussians), max_len), np.nan)
    for i, traj in enumerate(all_num_unique_gaussians):
        padded_data[i, :len(traj)] = traj
        
    # Calculate quantiles ignoring NaNs
    q25 = np.nanpercentile(padded_data, 25, axis=0)
    q50 = np.nanpercentile(padded_data, 50, axis=0) # median
    q75 = np.nanpercentile(padded_data, 75, axis=0)
    
    time_axis = np.arange(max_len)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_axis, q50, color='blue', linewidth=2, label='Median Unique Gaussians')
    ax.fill_between(time_axis, q25, q75, color='blue', alpha=0.2, label='25%-75% Quantile')
    
    ax.set_title("Unique Selected Gaussians over Time", fontsize=14, fontweight='bold')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of Unique Gaussians")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def apply_heatmap_to_full_scene(env_wrapper, pool_idx: torch.Tensor, obs_gs_rgb: torch.Tensor,
                                 bg_opacity: float = 0.1, highlight_opacity: float = 1.0, colormap: str = 'cool'):
    """
    Computes the heatmap for the 1024 sampled Gaussians and injects them back into the full scene.
    The rest of the scene remains its original color.
    
    Also overwrites opacities so that highlighted Gaussians are fully opaque
    and the rest of the scene is semi-transparent, preventing occlusion issues during rendering.
    
    Args:
        bg_opacity: opacity for non-highlighted Gaussians in [0, 1] (default 0.2)
        highlight_opacity: opacity for highlighted Gaussians in [0, 1] (default 1.0)
    """
    heatmap_rgb = compute_gs_heatmap(pool_idx, obs_gs_rgb, colormap=colormap)
    
    gs_world_wrapper = env_wrapper.env

    # Map the 1024 sampled points back to indices in the full Gaussian scene
    moving_to_full_indices = torch.cat([
        gs_world_wrapper._semantic_indices[k] 
        for k in gs_world_wrapper.moving_gaussians.keys() 
    ])
    full_indices_1024 = moving_to_full_indices[env_wrapper.gaussian_indices]

    # Overwrite colors at the highlighted indices
    gs_world_wrapper.overwrite_gs_rgb_for_rendering(heatmap_rgb, full_indices_1024)

    # Build full-scene opacity: background gets dimmed, highlighted Gaussians are fully opaque
    N_total = gs_world_wrapper.merged_init_gaussian_models._opacity.shape[0]
    device = full_indices_1024.device
    all_opacities = torch.full((N_total,), bg_opacity, device=device)
    all_opacities[full_indices_1024] = highlight_opacity
    gs_world_wrapper.overwrite_gs_opacity_for_rendering(all_opacities)


def compute_gs_heatmap(pool_indices: torch.Tensor, original_rgb: torch.Tensor, colormap='cool') -> torch.Tensor:
    """
    Args:
        pool_indices: (K) tensor from the max pool.
        original_rgb: (N, 3) tensor of the raw Gaussian colors [0, 1].
    Returns:
        heatmap_rgb: (N, 3) tensor ready for rendering.
    """
    N, _ = original_rgb.shape
    device = original_rgb.device
    heatmap_rgb = original_rgb.clone()

    cmap = plt.get_cmap(colormap)

    # 1. Count frequencies
    unique_idx, counts = torch.unique(pool_indices, return_counts=True)
    
    # 2. Normalize counts to [0, 1] for the colormap
    max_count = counts.max().float().clamp(min=1e-5)
    normalized_intensity = counts.float() / max_count
    
    # 3. Yellow tint for ignored Gaussians
    # Using standard luminance weights to get grayscale intensity
    gray = (0.299 * heatmap_rgb[:, 0] + 
            0.587 * heatmap_rgb[:, 1] + 
            0.114 * heatmap_rgb[:, 2])
    # Apply yellow tint (R=gray, G=gray, B=0)
    heatmap_rgb[:, 0] = gray
    heatmap_rgb[:, 1] = gray
    heatmap_rgb[:, 2] = 0.0
    
    # 4. Map the selected Gaussians to vibrant colors
    # cmap returns (R, G, B, A) in [0, 1]. We take RGB.
    colors_np = cmap(normalized_intensity.cpu().numpy())[:, :3] 
    colors_tensor = torch.from_numpy(colors_np).to(device, dtype=torch.float32)
    
    # 5. Overwrite the grayscale with the heatmap colors for the winning indices
    heatmap_rgb[unique_idx, :] = colors_tensor

    return heatmap_rgb


def add_legend_to_video(video_frames, colormap='cool'):
    """
    Appends a legend (colorbar) to the bottom of the video frames.
    video_frames: (T, H, W, C) numpy array of uint8
    """

    T, H, W, C = video_frames.shape
    dpi = 100
    fig_height = 1.0  # inches
    fig = Figure(figsize=(W / dpi, fig_height), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    
    # Create axes for colorbar
    ax = fig.add_axes([0.1, 0.4, 0.8, 0.2]) # left, bottom, width, height
    
    cmap = plt.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax, orientation='horizontal')
    cb.set_label('Policy Attention Frequency (Normalized)', fontsize=8)
    cb.ax.tick_params(labelsize=8)
    
    # Add descriptive text
    fig.text(0.5, 0.8, "Magenta: High Attention | Cyan: Low Attention | Yellow: Ignored", 
             ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Render to numpy array
    canvas.draw()
    colorbar_img = np.asarray(canvas.buffer_rgba())[..., :3] # Get RGB channels
    
    # Ensure width matches exactly
    ch, cw, _ = colorbar_img.shape
    if cw != W:
        colorbar_img = cv2.resize(colorbar_img, (W, ch))
        
    # Tile across all frames
    colorbar_video = np.tile(colorbar_img[None, ...], (T, 1, 1, 1))
    
    # Concatenate vertically
    return np.concatenate([video_frames, colorbar_video], axis=1)


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
        B = obs_dict['gs_positions'].shape[0]
        N = obs_dict['gs_positions'].shape[2]

        # Dynamically set dummy indices on GPU
        self.obs_encoder.extractor._latest_pool_indices = torch.arange(N, dtype=torch.long, device=self.device)

        # Return random dummy actions for sequence
        action = torch.rand((B, self.action_steps, self.action_dim), device=self.device)
        return {'action': action}


if __name__ == "__main__":
    import hydra
    import sys
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    # Set default task for this runner script if not passed
    if not any(arg.startswith("task=") for arg in sys.argv):
        sys.argv.append("task=maniskill_wrist_cam_gs_stack")

    @hydra.main(
        version_base=None,
        config_path="../config",
        config_name="wrist_cam_gsplat_dp3"
    )
    def main(cfg):
        device = "cuda:0"
        output_dir = "test_eval_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir)

        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)

        action_dim = 8 if env_runner.representation_space == "pd_joint_pos" else 10
        policy = DummyPolicy(
            action_dim=action_dim,
            action_steps=8,
            device=device
        )

        runner_log_train = env_runner.run(policy)
        print("Log data:", runner_log_train)
        
    main()
