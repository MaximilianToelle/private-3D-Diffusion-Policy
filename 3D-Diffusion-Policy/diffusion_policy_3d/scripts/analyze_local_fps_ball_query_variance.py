import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from diffusion_policy_3d.dataset.maniskill_wrist_cam_gs_dataset import WristCamGSManiskillDataset
from pytorch3d.ops import sample_farthest_points, ball_query


def main():
    zarr_path = "/home/max/projects/gsplat_policy/datasets/gsplat_3d_diffusion_policy/stack_wrist_cam_surface_normals_semantics"
    
    print("Initializing dataset...")
    # Initialize without limiting horizon to get the full episode easily
    dataset = WristCamGSManiskillDataset(
        zarr_path=zarr_path,
        horizon=1, 
        n_obs_steps=1,
    )
    
    print("Loading normalizer...")
    normalizer = dataset.get_normalizer()
    normalizer.eval()
    normalizer.to('cuda:0')
    
    # 1. Get a random init state from the dataset
    seed = 42
    n_episodes = dataset.replay_buffer.n_episodes
    rng = np.random.default_rng(seed=seed)
    train_episode_idx = rng.choice(n_episodes, size=1, replace=False)[0]
    
    print(f"Selected training episode index: {train_episode_idx} (out of {n_episodes})")

    if train_episode_idx == 0:
        start_idx = 0
    else:
        start_idx = dataset.replay_buffer.meta['episode_ends'][train_episode_idx - 1]
    
    # We want the FULL trajectory
    end_idx = dataset.replay_buffer.meta['episode_ends'][train_episode_idx]
    
    print(f"Trajectory length: {end_idx - start_idx}")
    
    sample = {
        'state': dataset.state_array[start_idx:end_idx],
        'action': dataset.replay_buffer.root['data']['action'][start_idx:end_idx],
        'gsplats': dataset.gsplats_array[start_idx:end_idx]
    }
    
    # Extract raw data (skip subsampling to get all Gaussians)
    print("Extracting raw point cloud sequence...")
    batch_orig = dataset._sample_to_data(sample, skip_subsampling=False)
    
    # Move to GPU and add batch dimension (B=1, T=traj_len, N, D)
    for k, v in batch_orig['obs'].items():
        if isinstance(v, torch.Tensor):
            batch_orig['obs'][k] = v.unsqueeze(0).cuda()
            
    obs = batch_orig['obs']
    B, T, N, _ = obs['gs_positions'].shape
    device = obs['gs_positions'].device
    
    print("Applying FPS and Ball Query independently at EACH timestep...")
    # We treat T as the batch dimension to process all timesteps in parallel
    points = obs['gs_positions'][0].to(torch.float32) # (T, N, 3)
    
    if 'gs_length' in obs:
        lengths_orig = obs['gs_length'][0] # shape (1,)
        lengths = lengths_orig.expand(T).to(torch.long)
    else:
        lengths = torch.full((T,), N, dtype=torch.long, device=device)
    
    # 1. FPS for all timesteps
    num_samples = 1024
    _, centroid_idx = sample_farthest_points(
        points, 
        lengths=lengths, 
        K=num_samples,
        random_start_point=False
    )
    # centroid_idx is (T, 1024)
    centroids = torch.gather(points, 1, centroid_idx.unsqueeze(-1).expand(-1, -1, 3)) # (T, 1024, 3)
    
    # 2. Ball Query for all timesteps
    max_neighbors = 128
    radius = 0.01
    lengths1 = torch.full((T,), num_samples, dtype=torch.long, device=device)
    dists, idxs, nn = ball_query(
        p1=centroids, 
        p2=points, 
        lengths1=lengths1,     
        lengths2=lengths,
        K=max_neighbors, 
        radius=radius
    )
    
    # idxs: (T, 1024, max_neighbors)
    valid_mask = (idxs != -1) # (T, 1024, max_neighbors)
    
    # We replace -1 with 0 so we can safely gather. We will mask out the 0s later.
    safe_idxs = idxs.clone()
    safe_idxs[~valid_mask] = 0
    
    keys_to_analyze = ['gs_positions', 'gs_surface_normals', 'gs_log_scales']
    
    print("Extracting features for centroids and all neighbors...")
    
    val_orig_unnorm = {}
    val_aug_unnorm = {}
    val_orig_norm = {}
    val_aug_norm = {}
    
    T_idx = torch.arange(T, device=device).view(T, 1, 1)
    
    for key in keys_to_analyze:
        x = obs[key][0].to(torch.float32) # (T, N, D)
        D = x.shape[-1]
        
        # Centroid features
        gather_orig = centroid_idx.unsqueeze(-1).expand(-1, -1, D)
        v_orig = torch.gather(x, dim=1, index=gather_orig) # (T, 1024, D)
        
        # Neighbor features
        # safe_idxs is (T, 1024, max_neighbors)
        v_neighbors = x[T_idx, safe_idxs, :] # (T, 1024, max_neighbors, D)
        
        val_orig_unnorm[key] = v_orig
        val_aug_unnorm[key] = v_neighbors
        
        # Normalize
        norm_module = normalizer[key].cuda()
        val_orig_norm[key] = norm_module(v_orig)
        
        v_neighbors_flat = v_neighbors.view(T, 1024 * max_neighbors, D)
        v_neighbors_norm_flat = norm_module(v_neighbors_flat)
        val_aug_norm[key] = v_neighbors_norm_flat.view(T, 1024, max_neighbors, D)

    print("\n" + "="*50)
    print("COMPUTING AUGMENTATION PERTURBATION (ALL VALID NEIGHBORS)")
    print("="*50)
    
    valid_counts = valid_mask.sum(dim=2).long() # (T, 1024)
    valid_counts_clamped = torch.clamp(valid_counts, min=1).unsqueeze(-1) # (T, 1024, 1)
    
    metrics = {
        'mean': {'physical': {}, 'normalized': {}},
        'max': {'physical': {}, 'normalized': {}}
    }
    
    def compute_mean_max(metric_val):
        if metric_val.dim() == 4:
            mask = valid_mask.unsqueeze(-1)
            mean_val = (metric_val * mask).sum(dim=2) / valid_counts_clamped # (T, 1024, D)
            masked_metric = torch.where(mask, metric_val, torch.tensor(-1.0, device=device))
            max_val = masked_metric.max(dim=2)[0] # (T, 1024, D)
            return mean_val, max_val
        else:
            mask = valid_mask
            mean_val = (metric_val * mask).sum(dim=2) / valid_counts_clamped.squeeze(-1) # (T, 1024)
            masked_metric = torch.where(mask, metric_val, torch.tensor(-1.0, device=device))
            max_val = masked_metric.max(dim=2)[0] # (T, 1024)
            return mean_val, max_val
    
    for key in keys_to_analyze:
        # PHYSICAL
        v_orig_unnorm = val_orig_unnorm[key].unsqueeze(2) # (T, 1024, 1, D)
        v_aug_unnorm = val_aug_unnorm[key] # (T, 1024, max_neighbors, D)
        
        if key == 'gs_positions':
            dist = torch.norm(v_aug_unnorm - v_orig_unnorm, dim=-1) * 1000 # mm
            metrics['mean']['physical'][key], metrics['max']['physical'][key] = compute_mean_max(dist)
        elif key == 'gs_surface_normals':
            n1 = torch.nn.functional.normalize(v_orig_unnorm, dim=-1)
            n2 = torch.nn.functional.normalize(v_aug_unnorm, dim=-1)
            dot_product = (n1 * n2).sum(dim=-1).clip(-1.0, 1.0)
            angle = torch.acos(dot_product) * (180.0 / np.pi)
            metrics['mean']['physical'][key], metrics['max']['physical'][key] = compute_mean_max(angle)
        elif key == 'gs_log_scales':
            log_scale_diff = torch.abs(v_aug_unnorm - v_orig_unnorm)
            ratio = torch.exp(log_scale_diff)
            mean_ratio, max_ratio = compute_mean_max(ratio)
            metrics['mean']['physical'][key] = mean_ratio.view(T, -1)
            metrics['max']['physical'][key] = max_ratio.view(T, -1)
            
        # NORMALIZED
        v_orig_norm = val_orig_norm[key].unsqueeze(2) # (T, 1024, 1, D)
        v_aug_norm = val_aug_norm[key] # (T, 1024, max_neighbors, D)
        
        if key == 'gs_positions':
            dist = torch.norm(v_aug_norm - v_orig_norm, dim=-1)
            metrics['mean']['normalized'][key], metrics['max']['normalized'][key] = compute_mean_max(dist)
        elif key == 'gs_surface_normals':
            n1_norm = torch.nn.functional.normalize(v_orig_norm, dim=-1)
            n2_norm = torch.nn.functional.normalize(v_aug_norm, dim=-1)
            dot_product_norm = (n1_norm * n2_norm).sum(dim=-1).clip(-1.0, 1.0)
            angle = torch.acos(dot_product_norm) * (180.0 / np.pi)
            metrics['mean']['normalized'][key], metrics['max']['normalized'][key] = compute_mean_max(angle)
        elif key == 'gs_log_scales':
            log_scale_diff_norm = torch.abs(v_aug_norm - v_orig_norm)
            ratio = torch.exp(log_scale_diff_norm)
            mean_ratio, max_ratio = compute_mean_max(ratio)
            metrics['mean']['normalized'][key] = mean_ratio.view(T, -1)
            metrics['max']['normalized'][key] = max_ratio.view(T, -1)

    colors = {'gs_positions': 'blue', 'gs_surface_normals': 'orange', 'gs_log_scales': 'green'}
    titles = {
        'gs_positions': ('Euclidean Displacement (mm)', 'L2 Distance'),
        'gs_surface_normals': ('Angular Difference (degrees)', 'Angular Difference (degrees)'),
        'gs_log_scales': ('Scale Multiplier Ratio', 'Scale Multiplier Ratio')
    }

    # 1. Generate Box Plots for t=120
    t_box = 120
    print(f"\nCreating box plots for t={t_box}...")
    
    for case_name in ['mean', 'max']:
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        fig.suptitle(f"Ball Query {case_name.capitalize()} Deviation (t={t_box} | Episode {train_episode_idx})", fontsize=16)
        
        for idx, key in enumerate(keys_to_analyze):
            color = colors[key]
            for col, perspective in enumerate(['physical', 'normalized']):
                ax = axes[idx, col]
                val = metrics[case_name][perspective][key][t_box].cpu().numpy()
                
                bp = ax.boxplot(val, patch_artist=True, vert=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.5)
                
                ax.set_title(f"{key.upper()} ({perspective.capitalize()} - {case_name.capitalize()} Case)")
                ax.set_ylabel(titles[key][col])
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_xticks([])
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ball_query_deviation_boxplots_{case_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved box plot to {plot_path}")

    # 2. Generate Line Plots over all Timesteps
    print("\nCreating line plots over all timesteps...")
    
    time_steps = np.arange(T)
    for case_name in ['mean', 'max']:
        fig, axes = plt.subplots(3, 2, figsize=(16, 18), sharex=True)
        fig.suptitle(f"Ball Query {case_name.capitalize()} Deviation Over Time (Episode {train_episode_idx}, Length {T})", fontsize=16)
        
        for idx, key in enumerate(keys_to_analyze):
            color = colors[key]
            for col, perspective in enumerate(['physical', 'normalized']):
                ax = axes[idx, col]
                val = metrics[case_name][perspective][key] # (T, N)
                
                median_t = torch.quantile(val, 0.5, dim=1).cpu().numpy()
                q25_t = torch.quantile(val, 0.25, dim=1).cpu().numpy()
                q75_t = torch.quantile(val, 0.75, dim=1).cpu().numpy()
                
                iqr = q75_t - q25_t
                theo_lower = q25_t - 1.5 * iqr
                theo_upper = q75_t + 1.5 * iqr
                
                # Compute true empirical whiskers (like matplotlib boxplot)
                val_np = val.cpu().numpy()
                lower_whisker = np.zeros(T)
                upper_whisker = np.zeros(T)
                
                for t in range(T):
                    valid_lower = val_np[t][val_np[t] >= theo_lower[t]]
                    lower_whisker[t] = valid_lower.min() if len(valid_lower) > 0 else theo_lower[t]
                    
                    valid_upper = val_np[t][val_np[t] <= theo_upper[t]]
                    upper_whisker[t] = valid_upper.max() if len(valid_upper) > 0 else theo_upper[t]
                
                ax.plot(time_steps, median_t, label='Median Deviation', color=color)
                ax.fill_between(time_steps, q25_t, q75_t, color=color, alpha=0.3, label='25th-75th Percentile')
                ax.plot(time_steps, lower_whisker, linestyle='--', color=color, alpha=0.7, label='Lower Whisker (1.5 IQR)')
                ax.plot(time_steps, upper_whisker, linestyle='--', color=color, alpha=0.7, label='Upper Whisker (1.5 IQR)')
                
                ax.set_title(f"{key.upper()} ({perspective.capitalize()} - {case_name.capitalize()} Case)")
                ax.set_ylabel(titles[key][col])
                ax.legend(loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.7)
                if idx == 2:
                    ax.set_xlabel('Timestep')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"trajectory_local_fps_ball_query_deviation_{case_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved line plot to {plot_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
