import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from diffusion_policy_3d.dataset.maniskill_wrist_cam_gs_dataset import WristCamGSManiskillDataset
from diffusion_policy_3d.dataset.data_augmentations import GaussianRobotPoseJitter, GaussianFPS
import copy


def main():
    # ----------------------------------------------------
    # Configuration Options
    # ----------------------------------------------------
    # 'full_scene_jitter': Skip subsampling to load the FULL raw scene of high opacity Gaussians. 
    #                      FPS is then applied to the full scene to compute jitter and visualize.
    # 'active_training': Do NOT skip subsampling. This filters for active Gaussians and downsamples 
    #                    to INTERMEDIATE_SIZE exactly as during training. FPS is then applied to compute and visualize.
    mode = 'active_training' # Options: 'full_scene_jitter', 'active_training'
    
    zarr_path = "/home/max/projects/gsplat_policy/datasets/gsplat_3d_diffusion_policy/stack_wrist_cam_surface_normals_semantics"
    urdf_path = "/home/max/projects/gsplat_policy/submodules/GSWorld/gsworld/mani_skill/assets/robots/panda/fr3_umi_wrist435_modified.urdf"
    
    print("Initializing dataset...")
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
    torch.manual_seed(seed)
    n_episodes = dataset.replay_buffer.n_episodes
    rng = np.random.default_rng(seed=seed)
    train_episode_idx = rng.choice(n_episodes, size=1, replace=False)[0]
    
    print(f"Selected training episode index: {train_episode_idx} (out of {n_episodes})")
 
    if train_episode_idx == 0:
        start_idx = 0
    else:
        start_idx = dataset.replay_buffer.meta['episode_ends'][train_episode_idx - 1]
    
    # We only need the very first step (init state)
    end_idx = start_idx + 1
    
    sample = {
        'state': dataset.state_array[start_idx+120:end_idx+120],
        'action': dataset.replay_buffer.root['data']['action'][start_idx+120:end_idx+120],
        'gsplats': dataset.gsplats_array[start_idx+120:end_idx+120]
    }
    
    # Extract data from dataset based on the chosen mode
    print(f"Extracting Gaussians from dataset (mode: {mode})...")
    skip_subsampling = (mode == 'full_scene_jitter')
    batch_orig = dataset._sample_to_data(sample, skip_subsampling=skip_subsampling)
    
    # Move to GPU and add batch dimension (since we bypassed dataloader)
    for k, v in batch_orig['obs'].items():
        if isinstance(v, torch.Tensor):
            batch_orig['obs'][k] = v.unsqueeze(0).cuda()
    batch_orig['action'] = batch_orig['action'].unsqueeze(0).cuda()        

    # 2. Apply jitter
    print("Applying GaussianRobotPoseJitter...")
    jitter_aug = GaussianRobotPoseJitter(
        urdf_path=urdf_path,
        base_limit=0.1,
        wrist_limit=0.1
    )
    
    batch_jittered = jitter_aug(copy.deepcopy(batch_orig))
    
    # Verify true physical shift applied
    max_physical_shift = torch.norm(batch_jittered['obs']['gs_positions'] - batch_orig['obs']['gs_positions'], dim=-1).max()
    print(f"TRUE Max physical shift applied to any Gaussian: {max_physical_shift.item() * 1000:.2f} mm")
    
    # 3. Compute FPS indices for both states, but apply them BOTH to the ORIGINAL batch
    print("Computing FPS indices for both states...")
    from pytorch3d.ops import sample_farthest_points
    
    points_orig = batch_orig['obs']['gs_positions'][:, 0, :, :].to(torch.float32)
    lengths_orig = batch_orig['obs'].get('gs_length', None)
    _, idx_orig = sample_farthest_points(
        points_orig, 
        lengths=lengths_orig, 
        K=1024, 
        random_start_point=False
    )
    
    points_jitter = batch_jittered['obs']['gs_positions'][:, 0, :, :].to(torch.float32)
    lengths_jitter = batch_jittered['obs'].get('gs_length', None)
    _, idx_jitter = sample_farthest_points(
        points_jitter, 
        lengths=lengths_jitter, 
        K=1024, 
        random_start_point=False
    )
    
    set_orig = set(idx_orig[0].cpu().numpy())
    set_jitter = set(idx_jitter[0].cpu().numpy())
    num_diff = len(set_orig - set_jitter)
    print(f"Number of different Gaussians selected by FPS due to jitter: {num_diff} / 1024")

    # Apply indices across all features to isolate the sampling variance
    obs_orig_sampled = {}
    obs_jitter_sampled = {}       # For the viewer (actual physically shifted points)
    obs_jitter_for_analysis = {}   # For the metrics (original physical points, jittered indices)
    
    B, T = batch_orig['obs']['gs_positions'].shape[:2]
    keys_to_gather = [k for k in batch_orig['obs'].keys() if k not in ['agent_pos', 'gs_length']]
    
    for key in keys_to_gather:
        tensor_orig = batch_orig['obs'][key]
        tensor_jitter = batch_jittered['obs'][key]
        D = tensor_orig.shape[-1]
        
        gather_idx_orig = idx_orig.view(B, 1, 1024, 1).expand(-1, T, -1, D)
        gather_idx_jitter = idx_jitter.view(B, 1, 1024, 1).expand(-1, T, -1, D)
        
        obs_orig_sampled[key] = torch.gather(tensor_orig, dim=2, index=gather_idx_orig).to(torch.float32)
        
        # For the viewer: the physically shifted points at the selected indices
        obs_jitter_sampled[key] = torch.gather(tensor_jitter, dim=2, index=gather_idx_jitter).to(torch.float32)
        
        # CRUCIAL: For metrics, we gather from tensor_orig using idx_jitter to isolate the FPS selection difference!
        obs_jitter_for_analysis[key] = torch.gather(tensor_orig, dim=2, index=gather_idx_jitter).to(torch.float32)
    
    keys_to_analyze = ['gs_positions', 'gs_surface_normals', 'gs_log_scales']
    
    # 4. Normalize
    print("Normalizing features...")
    norm_orig = {}
    norm_jitter = {}
    
    for key in keys_to_analyze:
        norm_module = normalizer[key].cuda()

        orig_val = obs_orig_sampled[key]
        jitter_val = obs_jitter_for_analysis[key]
        
        norm_orig[key] = norm_module(orig_val)
        norm_jitter[key] = norm_module(jitter_val)

    # 5. Compute the variance / absolute difference with Nearest Neighbor matching
    print("\n" + "="*50)
    print("ANALYSIS OF JITTER EFFECT AFTER DETERMINISTIC FPS (NN MATCHED)")
    print("="*50)
    
    # Use unnormalized positions to find the true spatial nearest neighbor
    pos_orig = obs_orig_sampled['gs_positions'].squeeze().to(torch.float32)
    pos_jitter = obs_jitter_for_analysis['gs_positions'].squeeze().to(torch.float32)
    
    # Compute pairwise Euclidean distances (1024 x 1024)
    dist = torch.cdist(pos_jitter.unsqueeze(0), pos_orig.unsqueeze(0)).squeeze(0)
    
    # For each jittered point, find the index of the closest original point
    nn_idx = torch.argmin(dist, dim=1)
    
    metrics = {
        'physical': {},
        'normalized': {}
    }
    
    for key in keys_to_analyze:
        # A. NETWORK PERCEPTION (Normalized Shift)
        val_orig_norm = norm_orig[key].squeeze() # (1024, D)
        val_jitter_norm = norm_jitter[key].squeeze() # (1024, D)
        val_orig_norm_matched = val_orig_norm[nn_idx]
        
        if key == 'gs_positions':
            metrics['normalized'][key] = torch.norm(val_jitter_norm - val_orig_norm_matched, dim=-1).cpu().numpy()
        elif key == 'gs_surface_normals':
            n1_norm = torch.nn.functional.normalize(val_orig_norm_matched, dim=-1)
            n2_norm = torch.nn.functional.normalize(val_jitter_norm, dim=-1)
            dot_product_norm = (n1_norm * n2_norm).sum(dim=-1).clip(-1.0, 1.0)
            metrics['normalized'][key] = (torch.acos(dot_product_norm) * (180.0 / np.pi)).cpu().numpy()
        elif key == 'gs_log_scales':
            log_scale_diff_norm = torch.abs(val_jitter_norm - val_orig_norm_matched)
            metrics['normalized'][key] = torch.exp(log_scale_diff_norm).flatten().cpu().numpy()

        # B. PHYSICAL MEANING (Unnormalized Shift)
        val_orig_unnorm = obs_orig_sampled[key].squeeze().to(torch.float32)[nn_idx]
        val_jitter_unnorm = obs_jitter_for_analysis[key].squeeze().to(torch.float32)
        
        if key == 'gs_positions':
            metrics['physical'][key] = (torch.norm(val_jitter_unnorm - val_orig_unnorm, dim=-1) * 1000).cpu().numpy() # mm
        elif key == 'gs_surface_normals':
            n1 = torch.nn.functional.normalize(val_orig_unnorm, dim=-1)
            n2 = torch.nn.functional.normalize(val_jitter_unnorm, dim=-1)
            dot_product = (n1 * n2).sum(dim=-1).clip(-1.0, 1.0)
            metrics['physical'][key] = (torch.acos(dot_product) * (180.0 / np.pi)).cpu().numpy()
        elif key == 'gs_log_scales':
            log_scale_diff = torch.abs(val_jitter_unnorm - val_orig_unnorm)
            metrics['physical'][key] = torch.exp(log_scale_diff).flatten().cpu().numpy()

    # Create box plots
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    colors = {'gs_positions': 'blue', 'gs_surface_normals': 'orange', 'gs_log_scales': 'green'}
    titles = {
        'gs_positions': ('Euclidean Displacement (mm)', 'L2 Distance'),
        'gs_surface_normals': ('Angular Difference (degrees)', 'Angular Difference (degrees)'),
        'gs_log_scales': ('Scale Multiplier Ratio', 'Scale Multiplier Ratio')
    }
    
    for idx, key in enumerate(keys_to_analyze):
        color = colors[key]
        for col, perspective in enumerate(['physical', 'normalized']):
            ax = axes[idx, col]
            val = metrics[perspective][key]
            
            bp = ax.boxplot(val, patch_artist=True, vert=True) # , whis=(0, 100))
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            
            ax.set_title(f"{key.upper()} ({perspective.capitalize()})")
            ax.set_ylabel(titles[key][col])
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticks([]) # Remove x-axis tick labels since it's just one box

    plt.tight_layout()
    jitter_plot_path = "/home/max/projects/gsplat_policy/jitter_sampling_difference_nn_matched_boxplots.png"
    plt.savefig(jitter_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nBox plots saved to {jitter_plot_path}")
        
    print("\nDone!")

    print("\nStarting 3D Viewer to visualize the physical jitter...")
    print("Open your browser to http://localhost:8081")
    
    try:
        from gsworld.mani_skill.utils.gsplat_viewer.gsplat_viewer import GsplatViewer
        from gsworld.mani_skill.utils.gsplat_viewer.utils_rasterize_render import _viewer_render_fn, _on_connect
        from diffusion_policy_3d.env.maniskill.maniskill_gs_wrapper import matrix_to_quaternion
        import viser
        from functools import partial
        import time
        
        from gsworld.constants import fr3_gs_semantics
        
        def prepare_gs_dict(obs):
            # Extract all Gaussians directly (no masking)
            positions = obs['gs_positions'].view(-1, 3).to(torch.float32).contiguous()
            rot_matrices = obs['gs_rotations_9d'].view(-1, 3, 3).to(torch.float32)
            quats = matrix_to_quaternion(rot_matrices).contiguous()
            scales = obs['gs_log_scales'].view(-1, 3).to(torch.float32).contiguous()
            
            # Avoid inf from logit(1.0) which crashes the CUDA rasterizer (causes huge blobs)
            opacities_raw = torch.clamp(obs['gs_opacities'].view(-1), 1e-4, 1.0 - 1e-4)
            logit_opacities = torch.logit(opacities_raw).to(torch.float32).contiguous()
            colors = obs['gs_rgb'].view(-1, 3).to(torch.float32).contiguous()
            
            print(f"Prepared {positions.shape[0]} Gaussians for the viewer.")
            print(f"Max linear scale (meters): {torch.exp(scales).max().item():.5f}")
            
            return {
                "means": positions,
                "quats": quats,
                "scales": scales,
                "rgb_colors": colors,
                "opacities": logit_opacities
            }
        
        # We visualize exactly what we computed metrics on (the 1024 FPS-sampled Gaussians)
        print(f"Preparing 1024 FPS sampled Gaussians ({mode}) for the viewer...")
        
        # Highlight the 0th Gaussian (the FPS starting point) to make it highly visible
        magenta = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32, device=obs_orig_sampled['gs_rgb'].device)
        obs_orig_sampled['gs_rgb'][:, :, 0, :] = magenta
        obs_orig_sampled['gs_log_scales'][:, :, 0, :] += 1.0  
        
        obs_jitter_sampled['gs_rgb'][:, :, 0, :] = magenta
        obs_jitter_sampled['gs_log_scales'][:, :, 0, :] += 1.0
        
        gs_orig = prepare_gs_dict(obs_orig_sampled)
        gs_jitter = prepare_gs_dict(obs_jitter_sampled)

        device = gs_orig["means"].device
        
        gs4viewer = {}
        for k in gs_orig.keys():
            gs4viewer[k] = gs_orig[k].clone()
            
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
        
        print("Toggling between ORIGINAL and JITTERED every 2 seconds... (Press Ctrl+C to stop)")
        state = 0
        while True:
            time.sleep(2.0)
            viewer.lock.acquire()
            source = gs_jitter if state == 0 else gs_orig
            for k in gs_orig.keys():
                gs4viewer[k] = source[k]
            state = 1 - state
            viewer.rerender(None)
            viewer.lock.release()
            viewer.update(0, 0)
            
    except ImportError as e:
        print(f"Could not load viewer dependencies: {e}")
    except KeyboardInterrupt:
        print("\nViewer closed.")

if __name__ == "__main__":
    main()
