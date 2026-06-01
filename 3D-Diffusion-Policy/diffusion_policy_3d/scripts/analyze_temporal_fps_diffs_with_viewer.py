import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial

from diffusion_policy_3d.dataset.maniskill_wrist_cam_gs_dataset import WristCamGSManiskillDataset
from pytorch3d.ops import sample_farthest_points

def main():
    zarr_path = "/home/max/projects/gsplat_policy/datasets/gsplat_3d_diffusion_policy/stack_wrist_cam_surface_normals_semantics"
    
    print("Initializing dataset...")
    dataset = WristCamGSManiskillDataset(
        zarr_path=zarr_path,
        horizon=1, 
        n_obs_steps=1,
    )
    
    print("Loading normalizer...")
    normalizer = dataset.get_normalizer()
    normalizer.eval()
    
    seed = 42
    torch.manual_seed(seed)
    n_episodes = dataset.replay_buffer.n_episodes
    rng = np.random.default_rng(seed=seed)
    train_episode_idx = rng.choice(n_episodes, size=1, replace=False)[0]
    
    print(f"Selected training episode index: {train_episode_idx} (out of {n_episodes})")

    if train_episode_idx == 0:
        start_idx = 0
    else:
        start_idx = dataset.replay_buffer.meta['episode_ends'][train_episode_idx - 1]
    end_idx = dataset.replay_buffer.meta['episode_ends'][train_episode_idx]
    T = end_idx - start_idx
    
    sample = {
        'state': dataset.state_array[start_idx:end_idx],
        'action': dataset.replay_buffer.root['data']['action'][start_idx:end_idx],
        'gsplats': dataset.gsplats_array[start_idx:end_idx]
    }
    
    # skip_subsampling=False to filter active Gaussians
    torch_data = dataset._sample_to_data(sample, skip_subsampling=False)
    obs = torch_data['obs']
    
    print(f"Trajectory length (T): {T}")
    
    # FPS
    print("Performing Farthest Point Sampling per timestep...")
    points = obs['gs_positions'].to(torch.float32).cuda()
    valid_length = obs['gs_length'].item() if obs['gs_length'].dim() == 0 else obs['gs_length'][0].item()
    lengths = torch.full((T,), valid_length, dtype=torch.long, device=points.device)
    
    K = 1024
    _, sampled_indices = sample_farthest_points(
        points, 
        lengths=lengths, 
        K=K,
        random_start_point=False
    ) # (T, K)
    
    # Calculate index differences
    diffs = []
    for t in range(1, T):
        set_t = set(sampled_indices[t].cpu().numpy())
        set_prev = set(sampled_indices[t-1].cpu().numpy())
        diff = len(set_t - set_prev)
        diffs.append(diff)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T), diffs, marker='.')
    plt.title('Number of differently selected FPS indices vs previous timestep')
    plt.xlabel('Timestep')
    plt.ylabel('Differing Indices (out of 1024)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = "/home/max/projects/gsplat_policy/temporal_fps_diffs.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Gather needed features for the viewer
    keys_to_gather = ['gs_positions', 'gs_rotations_9d', 'gs_log_scales', 'gs_opacities', 'gs_rgb', 'gs_surface_normals']
    sampled_obs = {}
    
    for key in keys_to_gather:
        if key not in obs: continue
        tensor = obs[key].cuda() # (T, N, D)
        D = tensor.shape[-1]
        gather_idx = sampled_indices.unsqueeze(-1).expand(-1, -1, D)
        sampled_obs[key] = torch.gather(tensor, dim=1, index=gather_idx).to(torch.float32)

    # ---------------------------------------------------------
    # Temporal FPS Difference Analysis (Nearest Neighbor Matched)
    # ---------------------------------------------------------
    # Apply selected indices across time to the *initial timestep* to isolate sampling variance
    keys_to_analyze = ['gs_positions', 'gs_surface_normals', 'gs_log_scales']
    
    sampled_obs_analysis = {}
    for key in keys_to_analyze:
        tensor_t0 = obs[key][0:1].cuda().expand(T, -1, -1) # (T, N, D) using the first timestep
        D = tensor_t0.shape[-1]
        gather_idx = sampled_indices.unsqueeze(-1).expand(-1, -1, D)
        sampled_obs_analysis[key] = torch.gather(tensor_t0, dim=1, index=gather_idx).to(torch.float32)
        
    normalized_obs_analysis = {}
    for key in keys_to_analyze:
        norm_module = normalizer[key].cuda()
        normalized_obs_analysis[key] = norm_module(sampled_obs_analysis[key])

    print("Computing Nearest Neighbor matching for temporal analysis...")
    pos_t = sampled_obs_analysis['gs_positions'] # (T, K, 3)
    pos_t0 = pos_t[0] # (K, 3)
    
    dist = torch.cdist(pos_t, pos_t0.unsqueeze(0).expand(T, -1, -1)) # (T, K, K)
    nn_idx = torch.argmin(dist, dim=2) # (T, K)
    
    # We will compute 6 metrics (3 physical, 3 normalized)
    metrics = {
        'physical': {},
        'normalized': {}
    }
    
    for key in keys_to_analyze:
        # PHYSICAL
        val_t0_unnorm = sampled_obs_analysis[key][0] # (K, D)
        val_t0_unnorm_matched = val_t0_unnorm[nn_idx] # (T, K, D)
        val_t_unnorm = sampled_obs_analysis[key] # (T, K, D)
        
        if key == 'gs_positions':
            dist_val = torch.norm(val_t_unnorm - val_t0_unnorm_matched, dim=-1) * 1000 # mm
            metrics['physical'][key] = dist_val
        elif key == 'gs_surface_normals':
            n1 = torch.nn.functional.normalize(val_t0_unnorm_matched, dim=-1)
            n2 = torch.nn.functional.normalize(val_t_unnorm, dim=-1)
            dot_product = (n1 * n2).sum(dim=-1).clip(-1.0, 1.0)
            metrics['physical'][key] = torch.acos(dot_product) * (180.0 / np.pi) # degrees
        elif key == 'gs_log_scales':
            log_scale_diff = torch.abs(val_t_unnorm - val_t0_unnorm_matched)
            metrics['physical'][key] = torch.exp(log_scale_diff).view(T, -1) # Flatten axes
            
        # NORMALIZED
        val_t0_norm = normalized_obs_analysis[key][0] # (K, D)
        val_t0_norm_matched = val_t0_norm[nn_idx] # (T, K, D)
        val_t_norm = normalized_obs_analysis[key] # (T, K, D)
        
        if key == 'gs_positions':
            metrics['normalized'][key] = torch.norm(val_t_norm - val_t0_norm_matched, dim=-1)
        elif key == 'gs_surface_normals':
            n1_norm = torch.nn.functional.normalize(val_t0_norm_matched, dim=-1)
            n2_norm = torch.nn.functional.normalize(val_t_norm, dim=-1)
            dot_product_norm = (n1_norm * n2_norm).sum(dim=-1).clip(-1.0, 1.0)
            metrics['normalized'][key] = torch.acos(dot_product_norm) * (180.0 / np.pi)
        elif key == 'gs_log_scales':
            log_scale_diff_norm = torch.abs(val_t_norm - val_t0_norm_matched)
            metrics['normalized'][key] = torch.exp(log_scale_diff_norm).view(T, -1)
            
    fig, axes = plt.subplots(3, 2, figsize=(16, 18), sharex=True)
    colors = {'gs_positions': 'blue', 'gs_surface_normals': 'orange', 'gs_log_scales': 'green'}
    titles = {
        'gs_positions': ('Euclidean Displacement (mm)', 'L2 Distance'),
        'gs_surface_normals': ('Angular Difference (degrees)', 'Angular Difference (degrees)'),
        'gs_log_scales': ('Scale Multiplier Ratio', 'Scale Multiplier Ratio')
    }
    
    x = np.arange(T)
    for idx, key in enumerate(keys_to_analyze):
        color = colors[key]
        for col, perspective in enumerate(['physical', 'normalized']):
            ax = axes[idx, col]
            val = metrics[perspective][key] # (T, K) or (T, K*3)
            
            median_t = torch.quantile(val, 0.5, dim=1).cpu().numpy()
            q25_t = torch.quantile(val, 0.25, dim=1).cpu().numpy()
            q75_t = torch.quantile(val, 0.75, dim=1).cpu().numpy()
            
            ax.plot(x, median_t, label='Median', color=color)
            ax.fill_between(x, q25_t, q75_t, color=color, alpha=0.3, label='25th-75th Percentile')
            
            ax.set_title(f"{key.upper()} ({perspective.capitalize()})")
            ax.set_ylabel(titles[key][col])
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.7)
            if idx == 2:
                ax.set_xlabel('Timestep')

    plt.tight_layout()
    variance_plot_path = "/home/max/projects/gsplat_policy/temporal_sampling_difference_nn_matched.png"
    plt.savefig(variance_plot_path, dpi=300, bbox_inches='tight')
    print(f"Variance plot saved to {variance_plot_path}")

    # Highlight 0th Gaussian across all timesteps for the viewer
    print("Highlighting the 0th FPS selected Gaussian in magenta...")
    magenta = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32, device=sampled_obs['gs_rgb'].device)
    sampled_obs['gs_rgb'][:, 0, :] = magenta
    sampled_obs['gs_log_scales'][:, 0, :] += 1.0

    # Viewer Setup
    print("\nStarting 3D Viewer to visualize the trajectory...")
    print("Open your browser to http://localhost:8081")
    
    try:
        from gsworld.mani_skill.utils.gsplat_viewer.gsplat_viewer import GsplatViewer
        from gsworld.mani_skill.utils.gsplat_viewer.utils_rasterize_render import _viewer_render_fn, _on_connect
        from diffusion_policy_3d.env.maniskill.maniskill_gs_wrapper import matrix_to_quaternion
        import viser
        
        def prepare_gs_dict_t(obs_t):
            positions = obs_t['gs_positions'].view(-1, 3).contiguous()
            rot_matrices = obs_t['gs_rotations_9d'].view(-1, 3, 3)
            quats = matrix_to_quaternion(rot_matrices).contiguous()
            scales = obs_t['gs_log_scales'].view(-1, 3).contiguous()
            
            opacities_raw = torch.clamp(obs_t['gs_opacities'].view(-1), 1e-4, 1.0 - 1e-4)
            logit_opacities = torch.logit(opacities_raw).contiguous()
            colors = obs_t['gs_rgb'].view(-1, 3).contiguous()
            
            return {
                "means": positions,
                "quats": quats,
                "scales": scales,
                "rgb_colors": colors,
                "opacities": logit_opacities
            }

        # Precompute dicts for all timesteps
        gs_dicts = []
        for t in range(T):
            obs_t = {k: sampled_obs[k][t] for k in keys_to_gather}
            gs_dicts.append(prepare_gs_dict_t(obs_t))
            
        device = gs_dicts[0]["means"].device
        
        gs4viewer = {}
        for k in gs_dicts[0].keys():
            gs4viewer[k] = gs_dicts[0][k].clone()
            
        scene_center = (gs4viewer["means"].data.mean(dim=0).cpu().numpy().tolist())
        
        server = viser.ViserServer(port=8081, verbose=False)
        viewer = GsplatViewer(
            server=server,
            render_fn=lambda camera_state, render_tab_state: _viewer_render_fn(
                camera_state, render_tab_state, gs4viewer, "3dgs", device
            ),
            output_dir=None,
            mode="training",
        )
        
        time.sleep(1)
        server.on_client_connect(partial(_on_connect, server=server, scene_center=scene_center))
        
        print("Playing through the trajectory... (Press Ctrl+C to stop)")
        t = 0
        while True:
            time.sleep(0.1) # 10 FPS
            viewer.lock.acquire()
            source = gs_dicts[t]
            for k in gs_dicts[0].keys():
                gs4viewer[k] = source[k]
            t = (t + 1) % T
            viewer.rerender(None)
            viewer.lock.release()
            viewer.update(0, 0)
            
    except ImportError as e:
        print(f"Could not load viewer dependencies: {e}")
    except KeyboardInterrupt:
        print("\nViewer closed.")

if __name__ == "__main__":
    main()
