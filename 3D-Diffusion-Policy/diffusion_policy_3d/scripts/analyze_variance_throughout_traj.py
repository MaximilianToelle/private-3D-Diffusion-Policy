import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from diffusion_policy_3d.dataset.maniskill_wrist_cam_gs_dataset import WristCamGSManiskillDataset
from pytorch3d.ops import sample_farthest_points


def main():
    zarr_path = "/home/max/projects/gsplat_policy/datasets/gsplat_3d_diffusion_policy/stack_wrist_cam_surface_normals"
    
    print("Initializing dataset...")
    dataset = WristCamGSManiskillDataset(
        zarr_path=zarr_path,
        horizon=1, 
        n_obs_steps=1,
    )
    
    print("Loading normalizer...")
    normalizer = dataset.get_normalizer()
    normalizer.eval()
    
    print("Extracting the correct trajectory...")
    seed = 42
    n_episodes = dataset.replay_buffer.n_episodes
    rng = np.random.default_rng(seed=seed)
    train_episode_idx = rng.choice(n_episodes, size=1, replace=False)[0]
    
    print(f"Selected training episode index: {train_episode_idx} (out of {n_episodes})")

    # Get the start and end indices of the selected episode
    if train_episode_idx == 0:
        start_idx = 0
    else:
        start_idx = dataset.replay_buffer.meta['episode_ends'][train_episode_idx - 1]
    end_idx = dataset.replay_buffer.meta['episode_ends'][train_episode_idx]
    T = end_idx - start_idx
    
    # Manually extract the first episode's raw data
    sample = {
        'state': dataset.state_array[start_idx:end_idx],
        'action': dataset.replay_buffer.root['data']['action'][start_idx:end_idx],
        'gsplats': dataset.gsplats_array[start_idx:end_idx]
    }
    
    # Use dataset's method to handle masking/padding and move to torch
    torch_data = dataset._sample_to_data(sample, skip_subsampling=False)
    obs = torch_data['obs']
    
    print(f"Trajectory length (T): {T}")
    
    # ---------------------------------------------------------
    # FPS per timestep
    # ---------------------------------------------------------
    print("Performing Farthest Point Sampling per timestep...")
    points = obs['gs_positions'].to(torch.float32).cuda()
    
    # obs['gs_length'] is a scalar tensor of the valid length (assuming skip_subsampling=False returns a single valid length)
    valid_length = obs['gs_length'].item()
    lengths = torch.full((T,), valid_length, dtype=torch.long, device=points.device)
    
    K = 1024
    _, sampled_indices = sample_farthest_points(
        points, 
        lengths=lengths, 
        K=K
    ) # sampled_indices: (T, K)
    
    keys_to_analyze = ['gs_positions', 'gs_surface_normals', 'gs_log_scales']
    sampled_obs = {}
    
    for key in keys_to_analyze:
        tensor = obs[key].cuda() # (T, N, D)
        D = tensor.shape[-1]
        gather_idx = sampled_indices.unsqueeze(-1).expand(-1, -1, D)
        sampled_obs[key] = torch.gather(tensor, dim=1, index=gather_idx).to(torch.float32)
        
    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------
    print("Normalizing features...")
    normalized_obs = {}
    for key in keys_to_analyze:
        norm_module = normalizer[key].cuda()
        normalized_obs[key] = norm_module(sampled_obs[key])
        
    normalized_agent_pos = normalizer['agent_pos'].cuda()(obs['agent_pos'].to(torch.float32).cuda())
        
    # ---------------------------------------------------------
    # Variance Analysis & Plotting
    # ---------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    colors = {'gs_positions': 'blue', 'gs_surface_normals': 'orange', 'gs_log_scales': 'green'}
    
    ax_combined = axes[3]
    
    for idx, key in enumerate(keys_to_analyze):
        tensor = normalized_obs[key] # (T, K, D)
        T_len = tensor.shape[0]
        
        # Spatial Variance across K points -> (T, D)
        var_spatial = tensor.var(dim=1) 
        
        # 3. For plotting: compute per-timestep median and percentiles across D dimensions
        median_t = torch.quantile(var_spatial, 0.5, dim=1).cpu().numpy()
        q25_t = torch.quantile(var_spatial, 0.25, dim=1).cpu().numpy()
        q75_t = torch.quantile(var_spatial, 0.75, dim=1).cpu().numpy()
        min_t = var_spatial.min(dim=1).values.cpu().numpy()
        max_t = var_spatial.max(dim=1).values.cpu().numpy()
        
        ax = axes[idx]
        color = colors[key]
        x = np.arange(T_len)
        
        # Individual subplot
        ax.plot(x, median_t, label=f'{key} (Median Var)', color=color)
        ax.fill_between(x, min_t, max_t, color=color, alpha=0.1, label='Min/Max Var')
        ax.fill_between(x, q25_t, q75_t, color=color, alpha=0.3, label='25th-75th Var')
        ax.set_title(f'{key.upper()} Spatial Variance over Trajectory')
        ax.set_ylabel('Variance')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Combined subplot
        ax_combined.plot(x, median_t, label=f'{key} (Median Var)', color=color)
        ax_combined.fill_between(x, min_t, max_t, color=color, alpha=0.1)
        ax_combined.fill_between(x, q25_t, q75_t, color=color, alpha=0.3)
        
    ax_combined.set_title('Combined Spatial Variance (Median & Min/Max & 25th-75th)')
    ax_combined.set_ylabel('Variance')
    ax_combined.set_xlabel('Timestep')
    ax_combined.legend(loc='upper right')
    ax_combined.grid(True, linestyle='--', alpha=0.7)
    
    # ---------------------------------------------------------
    # Policy Encoder Temporal Jitter Analysis
    # ---------------------------------------------------------
    from train import TrainDP3Workspace
    import sys
    
    print("\n" + "="*50)
    print("POLICY ENCODER TEMPORAL JITTER ANALYSIS")
    print("="*50)

    runs = {
        "GOOD (Zeroed Normals/Scales)": "/home/max/projects/gsplat_policy/trainings/maniskill_wrist_cam_gs_stack/wrist_cam_gsplat_dp3/one_traj_raw_concat_pos_zeroed_normals_and_scales_seed42_20260522_210229/0/checkpoints/latest.ckpt",
        "BAD (With Normals/Scales)": "/home/max/projects/gsplat_policy/trainings/maniskill_wrist_cam_gs_stack/wrist_cam_gsplat_dp3/one_traj_raw_param_cat_pos_normals_ordered_scales_seed42_20260520_191453/0/checkpoints/latest.ckpt"
    }

    jitter_data = {}

    for name, ckpt_path in runs.items():
        print(f"\nAnalyzing run: {name}")
        print(f"Loading checkpoint: {ckpt_path}")
        
        try:
            workspace = TrainDP3Workspace.create_from_checkpoint(ckpt_path)
            policy = workspace.model
            policy.eval()
            policy.cuda()
            
            # Prepare input for the encoder
            encoder_input = {
                'agent_pos': normalized_agent_pos
            }
            for key in keys_to_analyze:
                if name == "GOOD (Zeroed Normals/Scales)" and key in ['gs_surface_normals', 'gs_log_scales']:
                    encoder_input[key] = torch.zeros_like(normalized_obs[key])
                else:
                    encoder_input[key] = normalized_obs[key]

            with torch.no_grad():
                global_feat = policy.obs_encoder(encoder_input) 

            # 1. Calculate the difference between consecutive timesteps
            # Shape becomes (T-1, feat_dim)
            step_diffs = torch.diff(global_feat, dim=0) 

            # 2. Calculate the absolute magnitude of these jumps
            abs_jumps = step_diffs.abs()

            # Store for plotting later
            jitter_data[name] = abs_jumps.cpu().numpy()

        except Exception as e:
            print(f"  Failed to evaluate: {e}")

    # First figure (Spatial Variance)
    plt.tight_layout()
    
    # Save to the artifacts directory
    output_path = "/home/max/projects/gsplat_policy/variance_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")

    # Second figure (Jitter Analysis)
    if jitter_data:
        fig_jitter, axes_jitter = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        colors = {
            "GOOD (Zeroed Normals/Scales)": "blue",
            "BAD (With Normals/Scales)": "red"
        }
        
        for name, abs_jumps_np in jitter_data.items():
            color = colors.get(name, "black")
            T_minus_1 = abs_jumps_np.shape[0]
            feat_dim = abs_jumps_np.shape[1]
            x = np.arange(T_minus_1)
            
            # We assume first 64 channels are gsplat features and next 64 are state features.
            
            # Gaussian Features (channels 0-63)
            if feat_dim > 0:
                gauss_jumps = abs_jumps_np[:, :min(64, feat_dim)]
                gauss_median = np.median(gauss_jumps, axis=1)
                gauss_q25 = np.percentile(gauss_jumps, 25, axis=1)
                gauss_q75 = np.percentile(gauss_jumps, 75, axis=1)
                
                axes_jitter[0].plot(x, gauss_median, color=color, label=f"{name} (Median)")
                axes_jitter[0].fill_between(x, gauss_q25, gauss_q75, color=color, alpha=0.2, label=f"{name} (25th-75th)")
            
            # State Features (channels 64-127)
            if feat_dim > 64:
                state_jumps = abs_jumps_np[:, 64:min(128, feat_dim)]
                state_median = np.median(state_jumps, axis=1)
                state_q25 = np.percentile(state_jumps, 25, axis=1)
                state_q75 = np.percentile(state_jumps, 75, axis=1)
                
                axes_jitter[1].plot(x, state_median, color=color, label=f"{name} (Median)")
                axes_jitter[1].fill_between(x, state_q25, state_q75, color=color, alpha=0.2, label=f"{name} (25th-75th)")
                
        axes_jitter[0].set_title('Gaussian Features Absolute Jitter Over Time (First 64 Channels)')
        axes_jitter[0].set_ylabel('Absolute Diff')
        axes_jitter[0].legend(loc='upper right')
        axes_jitter[0].grid(True, linestyle='--', alpha=0.7)
        
        axes_jitter[1].set_title('State Features Absolute Jitter Over Time (Next 64 Channels)')
        axes_jitter[1].set_ylabel('Absolute Diff')
        axes_jitter[1].set_xlabel('Timestep')
        axes_jitter[1].legend(loc='upper right')
        axes_jitter[1].grid(True, linestyle='--', alpha=0.7)
        
        fig_jitter.tight_layout()
        jitter_output_path = "/home/max/projects/gsplat_policy/jitter_plot.png"
        fig_jitter.savefig(jitter_output_path, dpi=300, bbox_inches='tight')
        print(f"Jitter plot saved to {jitter_output_path}")

if __name__ == "__main__":
    main()
