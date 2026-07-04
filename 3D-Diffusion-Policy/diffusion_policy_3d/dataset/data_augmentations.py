import torch
from pytorch3d.ops import sample_farthest_points, ball_query
from gsworld.constants import fr3_gs_semantics


class GaussianCompose:
    """Composes several transforms together sequentially."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, batch):
        for t in self.transforms:
            batch = t(batch)
        return batch


class GaussianFPS:
    """
    Standard Farthest Point Sampling on the GPU.
    Used for evaluation and baseline models to reduce the number of Gaussians.
    """
    def __init__(self, num_samples=1024, random_start_point=False):
        self.num_samples = num_samples
        self.random_start_point = random_start_point

    def __call__(self, batch):
        obs = batch['obs']
        B, T, N, _ = obs['gs_positions'].shape
        
        # Grab t=0 positions and the lengths tensor
        points_t0 = obs['gs_positions'][:, 0, :, :].to(torch.float32)
        lengths = obs.get('gs_length', None)
        if 'gs_length' in obs:
            del batch['obs']['gs_length']       # not needed afterwards
        
        # Batched GPU FPS with strict boundary enforcement
        _, sampled_indices = sample_farthest_points(
            points_t0, 
            lengths=lengths, 
            K=self.num_samples,
            random_start_point=self.random_start_point
        )
        
        keys_to_subsample = [k for k in obs.keys() if k not in ['agent_proprio', 'gs_length']]
            
        # Apply indices across all features and time steps
        for key in keys_to_subsample:
            tensor = obs[key]       # (B, T, 32768, D)
            D = tensor.shape[-1]
            
            # Expand indices from (B, num_samples) to (B, T, num_samples, D)
            gather_idx = sampled_indices.view(B, 1, self.num_samples, 1).expand(-1, T, -1, D)
            
            # Subsample to final target size
            obs[key] = torch.gather(tensor, dim=2, index=gather_idx).to(torch.float32)
            
        return batch


class HighOpacityActiveGaussianRandomSampling:
    """
    Filters out all non-active and low-opacity Gaussians.  
    Randomly samples num_samples of the remaining Gaussians on the GPU for each sequence in a batch.
    
    Ensures that only valid (non-padded) Gaussians are sampled using the lengths tensor (gs_length).
    The sampled indices are applied consistently across all timesteps and features.
    """
    def __init__(self, num_samples=1024, min_opacity=0.9):
        self.num_samples = num_samples
        self.min_opacity = min_opacity

        self.keys_not_to_subsample = ['agent_proprio', 'gs_length']

    def __call__(self, batch):
        obs = batch['obs']
        B, T, N, _ = obs['gs_positions'].shape
        device = obs['gs_positions'].device

        # Random sampling WITHOUT replacement using partial sort via topk
        r = torch.rand(B, N, device=device)
        
        # Initialize an all-True mask of shape (B, N)
        valid_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        
        if 'gs_opacities' in obs:
            # Handle potential time dimension T in gs_opacities (e.g. B, T, N, 1 or B, N, 1)
            opacities = obs['gs_opacities']
            if opacities.dim() == 4:
                opacities = opacities.min(dim=1)[0]  # Min over time
            high_opacity_gs_mask = (opacities >= self.min_opacity).squeeze(-1)
            valid_mask &= high_opacity_gs_mask
            del obs['gs_opacities']

        if 'gs_active_gaussians_mask' in obs:
            # Shape is (B, T, N, 1). Take min over T to see if it's active over the full observation window
            active_mask = obs['gs_active_gaussians_mask']
            if active_mask.dim() == 4:
                active_mask = active_mask.min(dim=1)[0]
            is_active_all_time = (active_mask > 0.5).squeeze(-1)
            valid_mask &= is_active_all_time
            del obs['gs_active_gaussians_mask']
        
        # Push invalid Gaussians to the back
        r[~valid_mask] = 2.0 
        
        assert (valid_mask.sum(dim=1) >= self.num_samples).all(), "Not enough valid Gaussians to sample from!"

        # topk(largest=False) returns K smallest values — i.e. K random valid indices
        _, sampled_indices = torch.topk(r, self.num_samples, dim=1, largest=False)
        
        keys_to_subsample = [k for k in obs.keys() if k not in self.keys_not_to_subsample]
            
        # Apply indices across all features and time steps
        for key in keys_to_subsample:
            tensor = obs[key]       # (B, T, N, D)
            D = tensor.shape[-1]
            
            # Expand indices from (B, num_samples) to (B, T, num_samples, D)
            gather_idx = sampled_indices.view(B, 1, self.num_samples, 1).expand(-1, T, -1, D)
            
            # Subsample to final target size
            obs[key] = torch.gather(tensor, dim=2, index=gather_idx).to(torch.float32)
            
        return batch


class GaussianFPSAndBallQuery:
    """
    Augmentation that runs FPS, then for each centroid, randomly selects
    one neighboring Gaussian within a specified radius using a ball query.
    """
    def __init__(self, num_samples=1024, random_start_point=False, radius=0.02, max_neighbors=16):
        self.num_samples = num_samples
        self.random_start_point = random_start_point
        self.radius = radius
        self.max_neighbors = max_neighbors

    def __call__(self, batch):
        obs = batch['obs']
        B, T, N, _ = obs['gs_positions'].shape
        device = obs['gs_positions'].device
        
        # Grab t=0 positions and the lengths tensor
        points_t0 = obs['gs_positions'][:, 0, :, :].to(torch.float32)
        lengths = obs.get('gs_length', None)
        if 'gs_length' in obs:
            del batch['obs']['gs_length']
            
        # 1. FPS to get centroids
        _, centroid_idx = sample_farthest_points(
            points_t0, 
            lengths=lengths, 
            K=self.num_samples,
            random_start_point=self.random_start_point
        )
        
        # Extract the centroid coordinates
        # Expand centroid_idx to (B, num_samples, 3) to gather from points_t0
        centroids = torch.gather(points_t0, 1, centroid_idx.unsqueeze(-1).expand(-1, -1, 3))
        
        # 2. Ball query around centroids
        # How many non-padded Gaussians are in the query space for each batch -> no padding
        lengths1 = torch.full((B,), self.num_samples, dtype=torch.long, device=points_t0.device)
        dists, idxs, nn = ball_query(
            p1=centroids, 
            p2=points_t0, 
            lengths1=lengths1,     
            lengths2=lengths,      # how many non-padded Gaussians are in the search space for each batch
            K=self.max_neighbors, 
            radius=self.radius
        )
        
        # 3. Randomly select one valid neighbor for each centroid
        # Resolve any collisions in a single pass using GPU scatter_
        valid_mask = idxs != -1
        valid_counts = valid_mask.sum(dim=-1) # shape: (B, num_samples)
        
        # Propose one (valid) random neighbor for each centroid
        # NOTE: proposed can still contain padded values if ball query did not find any neighbor!
        rand_cols = torch.randint(0, self.max_neighbors, (B, self.num_samples), device=device)
        rand_cols = torch.remainder(rand_cols, torch.clamp(valid_counts, min=1))    
        proposed = torch.gather(idxs, 2, rand_cols.unsqueeze(-1)).squeeze(-1) # (B, self.num_samples)
        
        # Resolve collisions directly among the proposed indices
        # matches[b, i, j] is True if centroid i and centroid j proposed the same index
        matches = proposed.unsqueeze(2) == proposed.unsqueeze(1) # (B, S, S)
        
        # We only look at previous proposals to find duplicates (lower triangle)
        # If a centroid's proposal matched any previous centroid's proposal, it's a loser
        prev_matches = torch.tril(matches, diagonal=-1)
        is_loser = prev_matches.any(dim=2)
        
        # A centroid is a winner if it is NOT a loser AND it didn't propose a -1 padding
        is_winner = (~is_loser) & (proposed != -1)
        
        # Winners keep their proposed neighbor; losers fall back to their FPS centroid
        sampled_indices = torch.where(is_winner, proposed, centroid_idx)
        
        keys_to_subsample = [k for k in obs.keys() if k not in ['agent_proprio', 'gs_length']]
        
        # Apply indices across all features and time steps
        for key in keys_to_subsample:
            tensor = obs[key]
            D = tensor.shape[-1]
            
            # Expand indices from (B, num_samples) to (B, T, num_samples, D)
            # This ensures EXACTLY the same Gaussians are selected across all timesteps T
            gather_idx = sampled_indices.view(B, 1, self.num_samples, 1).expand(-1, T, -1, D)
            
            # Subsample to final target size
            obs[key] = torch.gather(tensor, dim=2, index=gather_idx).to(torch.float32)
            
        return batch


class GaussianRobotPoseJitter:
    """
    TODO BIG PROBLEM: As soon as the robot is close to an object or grasped an object, this data augmentation creates physically non-plausible states!
    For example: 
        - The can is lifted up in the air, while the robot gripper is not touching the can anymore. 
        - The robot gripper was already close to the can, jitter moved it inside the can.
    """

    def __init__(self, urdf_path, base_limit=0.005, wrist_limit=0.01):
        raise NotImplementedError("Implementation is not feasible and not updated to latest obs keys")
        
        self.urdf_path = urdf_path
        self.base_limit = base_limit
        self.wrist_limit = wrist_limit
        
        # Load URDF via pytorch_kinematics
        import pytorch_kinematics as pk
        with open(urdf_path, 'r') as f:
            urdf_str = f.read()
        self.kinematic_chain = pk.build_chain_from_urdf(urdf_str)
        
        # Parse limits directly from URDF XML
        import xml.etree.ElementTree as ET
        root = ET.fromstring(urdf_str)
        
        self.joint_limits = {}
        for joint in root.findall('.//joint'):
            name = joint.get('name')
            limit_elem = joint.find('limit')
            if limit_elem is not None:
                lower = float(limit_elem.get('lower', -1e9))
                upper = float(limit_elem.get('upper', 1e9))
                self.joint_limits[name] = (lower, upper)
        
        # PyTorch Kinematics order of joints for this chain
        self.pk_joint_names = self.kinematic_chain.get_joint_parameter_names()
        
        # Pre-compute limits as tensors
        self.limits_lower = torch.tensor([self.joint_limits.get(name, (-1e9, 1e9))[0] for name in self.pk_joint_names], dtype=torch.float32)
        self.limits_upper = torch.tensor([self.joint_limits.get(name, (-1e9, 1e9))[1] for name in self.pk_joint_names], dtype=torch.float32)

        # Add small margin to not actually get into physical limits due to jitter
        margin = 0.01
        self.limits_lower[:7] = self.limits_lower[:7] + margin
        self.limits_upper[:7] = self.limits_upper[:7] - margin

    def apply_capped_qpos_jitter(self, agent_qpos):
        B, T, D = agent_qpos.shape
        device = agent_qpos.device
        
        # Ensure our limit tensors are on the correct device
        self.limits_lower = self.limits_lower.to(device)
        self.limits_upper = self.limits_upper.to(device)
        
        # 1. Generate Uniform Noise between [0, 1) constant over time
        noise_uniform = torch.rand(B, 1, 7, device=device)
        
        # 2. Scale to [-1, 1]
        noise_scaled = (noise_uniform * 2.0) - 1.0
        
        # 3. Apply specific limits to specific FR3 joints
        jitter_amounts = torch.zeros_like(noise_scaled)
        # Joints 0-3 (Base/Shoulder/Elbow) get a tighter limit
        jitter_amounts[..., 0:4] = noise_scaled[..., 0:4] * self.base_limit
        # Joints 4-6 (Wrist) get a slightly looser limit
        jitter_amounts[..., 4:7] = noise_scaled[..., 4:7] * self.wrist_limit
        
        jittered_qpos = agent_qpos.clone()
        jittered_qpos[..., :7] += jitter_amounts
        
        # 4. Clamp to respect physical joint limits
        jittered_qpos = torch.clamp(jittered_qpos, min=self.limits_lower, max=self.limits_upper)
        
        return jittered_qpos

    def __call__(self, batch):
        if self.base_limit == 0.0 and self.wrist_limit == 0.0:
            print("Returning batch without GaussianRobotPoseJitter Augmentation")
            return batch
            
        obs = batch['obs']
        if 'gs_semantics' not in obs or 'agent_proprio' not in obs:
            print("Returning batch without GaussianRobotPoseJitter Augmentation")
            return batch
            
        B, T, N, _ = obs['gs_positions'].shape
        device = obs['gs_positions'].device
        
        agent_qpos = obs['agent_proprio'] # (B, T, 9)
        jittered_qpos = self.apply_capped_qpos_jitter(agent_qpos)
        
        # Pass through PK (batch size B*T)
        self.kinematic_chain = self.kinematic_chain.to(device=device)
        qpos_flat = agent_qpos.view(B*T, -1)
        jitter_flat = jittered_qpos.view(B*T, -1)
        
        orig_transforms = self.kinematic_chain.forward_kinematics(qpos_flat)
        jitter_transforms = self.kinematic_chain.forward_kinematics(jitter_flat)
        
        for link_name in orig_transforms.keys():
            if link_name not in fr3_gs_semantics:
                print(f"{link_name} not in fr3_gs_semantics! Jitter is not being applied to corresponding Gaussians")
                continue
                
            sem_id = fr3_gs_semantics[link_name]

            # Create boolean mask for this link (B, T, N, 1)
            mask = (obs['gs_semantics'] == sem_id)
            
            # Only continue if there are any Gaussians for this link
            if not mask.any():
                continue
            
            # T_orig and T_jittered have shape (B*T, 4, 4)
            T_orig = orig_transforms[link_name].get_matrix()
            T_jittered = jitter_transforms[link_name].get_matrix()
            
            # Compute relative transform T_rel = T_jittered @ T_orig^-1
            T_orig_inv = torch.linalg.inv(T_orig)
            T_rel = torch.matmul(T_jittered, T_orig_inv)
            
            # Reshape to (B, T, 4, 4)
            T_rel = T_rel.view(B, T, 4, 4)
            
            # to apply to all Gaussians belonging to the link
            R_rel = T_rel[:, :, :3, :3].view(B, T, 1, 3, 3)
            t_rel = T_rel[:, :, :3, 3].view(B, T, 1, 3)
            
            # NOTE: We can apply the transformation because the robot base frame is aligned with sapien world frame where the Gaussians live
            # Apply to positions
            pos = obs['gs_positions'].to(torch.float32)
            perturbed_pos = torch.matmul(R_rel, pos.unsqueeze(-1)).squeeze(-1) + t_rel
            obs['gs_positions'] = torch.where(mask, perturbed_pos, pos).to(torch.float32)
            
            if 'point_cloud' in obs:
                obs['point_cloud'] = obs['gs_positions']
                
            if 'gs_surface_normals' in obs:
                normals = obs['gs_surface_normals'].to(torch.float32)
                perturbed_normals = torch.matmul(R_rel, normals.unsqueeze(-1)).squeeze(-1)
                obs['gs_surface_normals'] = torch.where(mask, perturbed_normals, normals).to(torch.float32)
                
            if 'gs_rotations_9d' in obs:
                rot_3x3 = obs['gs_rotations_9d'].view(B, T, N, 3, 3).to(torch.float32)
                perturbed_rot = torch.matmul(R_rel, rot_3x3)
                # mask is (B, T, N, 1), unsqueeze to broadcast to (B, T, N, 3, 3)
                obs['gs_rotations_9d'] = torch.where(mask.unsqueeze(-1), perturbed_rot, rot_3x3).view(B, T, N, 9).to(torch.float32)
                
        # Update agent_proprio to the jittered one
        obs['agent_proprio'] = jittered_qpos.to(torch.float32)
        
        return batch
