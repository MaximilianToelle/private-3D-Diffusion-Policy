import os
from typing import Dict
import torch
import numpy as np
import time
from tqdm import tqdm
import copy
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

from pytorch3d.ops import sample_farthest_points


class WristCamGSManiskillDataset(BaseDataset):
    INTERMEDIATE_SIZE = 32768   # doing farthest point sampling on GPU after batch generation
    
    def __init__(
        self,
        zarr_path,
        horizon=1,
        n_obs_steps=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        num_gaussians=1024,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.n_obs_steps = n_obs_steps

        self.replay_buffer = ReplayBuffer.create_from_path(
            zarr_path, mode='r')

        self.actor_keys = [k for k in self.replay_buffer.keys() if k.startswith('actor_pose_')]
        self.full_seq_keys = ['action'] + self.actor_keys
        
        # Policy only takes the first two obs out of the sequence, so we do manual slicing
        self.gsplats_array = self.replay_buffer.root['data']['gsplats']
        self.state_array = self.replay_buffer.root['data']['state']

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
            
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            keys=self.full_seq_keys
        )
            
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            keys=self.full_seq_keys,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs):
        stats_path = os.path.join(self.zarr_path, 'normalization_stats.pth')
        if os.path.exists(stats_path):
            print(f"Loading normalization stats from {stats_path}")
            state_dict = torch.load(stats_path)
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(state_dict)
            # for training dp3 on gsplat dataset
            if 'point_cloud' not in normalizer.params_dict and 'gs_positions' in normalizer.params_dict:
                normalizer['point_cloud'] = normalizer['gs_positions']
            if 'gs_surface_normals' not in normalizer.params_dict:
                normalizer['gs_surface_normals'] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)
            normalizer.to(torch.float32)
            return normalizer

        print(f"Dataset does not contain normalization stats yet. Stats are computed now and saved for future runs...")
        
        norm_mask = np.ones(self.replay_buffer.n_episodes, dtype=bool)
        norm_keys = ['action', 'state', 'gsplats']   # actor poses are only used for reproducing init states
        normalization_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=1,
            pad_before=0, 
            pad_after=0,
            episode_mask=norm_mask,
            keys=norm_keys
        )

        # Stride by sequence length to avoid overlapping frame duplication
        seq_length = getattr(normalization_sampler, 'sequence_length')
        indices = range(len(normalization_sampler))[::seq_length]
        
        # 1. Initialize running trackers for our statistics
        stats = {
            'action': {'min': None, 'max': None},
            'agent_pos': {'min': None, 'max': None},
            'gs_positions': {'min': None, 'max': None},
            # For Gaussian normalization, we need mean and std.
            # We track count, sum, and sum of squares (using float64 to prevent numerical instability)
            'gs_log_scales': {'count': 0, 'sum': 0.0, 'sum_sq': 0.0} 
        }

        # Helper function to update min/max dynamically
        def update_min_max(key, tensor):
            # Flatten everything except the last feature dimension
            flat_tensor = tensor.reshape(-1, tensor.shape[-1])
            batch_min = flat_tensor.min(dim=0)[0]
            batch_max = flat_tensor.max(dim=0)[0]
            
            if stats[key]['min'] is None:
                stats[key]['min'] = batch_min
                stats[key]['max'] = batch_max
            else:
                stats[key]['min'] = torch.minimum(stats[key]['min'], batch_min)
                stats[key]['max'] = torch.maximum(stats[key]['max'], batch_max)

        # 2. Streaming 
        for idx in tqdm(indices, desc="Streaming dataset for normalization stats"):
            sample = normalization_sampler.sample_sequence(idx)
            torch_data = self._sample_to_data(sample, skip_subsampling=True)    # IMPORTANT to see full GS scene during normalization
            obs = torch_data['obs']
            
            # Update min/max bounds
            update_min_max('action', torch_data['action'])
            update_min_max('agent_pos', obs['agent_pos'])
            update_min_max('gs_positions', obs['gs_positions'])
            
            # Update sum and sum of squares for log_scales
            # NOTE: We compute a single, global mean and std across scaling dimensions
            # -> A Gaussian can rotate 90 degree and swap its x- and y-scales without changing its appearance!   
            # NOTE: We cast to .double() to prevent catastrophic cancellation in large sum operations
            log_scales_flat = obs['gs_log_scales'].reshape(-1).double()
            stats['gs_log_scales']['count'] += log_scales_flat.shape[0]
            stats['gs_log_scales']['sum'] += log_scales_flat.sum(dim=0)
            stats['gs_log_scales']['sum_sq'] += (log_scales_flat ** 2).sum(dim=0)

        # 3. Compute final statistics from trackers
        normalizer = LinearNormalizer()

        # --- Positions (Preserving 3D Aspect Ratio) ---
        pos_min = stats['gs_positions']['min']
        pos_max = stats['gs_positions']['max']
        geometric_center = (pos_max + pos_min) / 2.0
        max_radius = torch.clamp((pos_max - pos_min).max() / 2.0, min=1e-4)

        normalizer['gs_positions'] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.ones_like(geometric_center) / max_radius,
            offset=-geometric_center / max_radius,
            input_stats_dict={
                'min': geometric_center - max_radius, 'max': geometric_center + max_radius,
                'mean': geometric_center, 'std': pos_max - pos_min
            }
        )
        normalizer['point_cloud'] = normalizer['gs_positions']

        # --- Log Scales (Gaussian Normalization) ---
        N = stats['gs_log_scales']['count']
        mean_log_scales = stats['gs_log_scales']['sum'] / N
        mean_squared_log_scales = stats['gs_log_scales']['sum_sq'] / N
        # Variance = (Sum of Squares / N) - (Mean ^ 2)
        var_log_scales = mean_squared_log_scales - (mean_log_scales ** 2)
        std_log_scales = torch.sqrt(torch.clamp(var_log_scales.float(), min=1e-6))
        # Cast to float32 after the math is safe
        mean_log_scales = mean_log_scales.float()

        normalizer['gs_log_scales'] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.full((3,), 1.0 / std_log_scales.item(), dtype=torch.float32),
            offset=torch.full((3,), -mean_log_scales.item() / std_log_scales.item(), dtype=torch.float32),
            input_stats_dict={
                'min': torch.full((3,), mean_log_scales.item() - 3*std_log_scales.item()), 
                'max': torch.full((3,), mean_log_scales.item() + 3*std_log_scales.item()),
                'mean': torch.full((3,), mean_log_scales.item()), 
                'std': torch.full((3,), std_log_scales.item())
            }
        )

        # --- Hardcoded Physics Bounds ---
        # by construction (orthogonal matrix) between [-1, 1]
        normalizer['gs_rotations_9d'] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)
        normalizer['gs_surface_normals'] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)

        # got processed with sigmoid -> [0, 1]
        normalizer['gs_opacities'] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.tensor([2.0], dtype=torch.float32), 
            offset=torch.tensor([-1.0], dtype=torch.float32),
            input_stats_dict={
                'min': torch.tensor([0.0]), 'max': torch.tensor([1.0]),
                'mean': torch.tensor([0.5]), 'std': torch.tensor([0.5])
            }
        )

        # got processed to be normalized between [0, 1]
        normalizer['gs_rgb'] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32), 
            offset=torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32),
            input_stats_dict={
                'min': torch.zeros(3), 'max': torch.ones(3),
                'mean': torch.full((3,), 0.5), 'std': torch.full((3,), 0.5)
            }
        )

        # --- Actions and Agent Pos (separate normalization of each DOF as they have individual physical ranges!) ---
        for key in ['action', 'agent_pos']:
            k_min = stats[key]['min']
            k_max = stats[key]['max']
            k_scale = torch.clamp(k_max - k_min, min=1e-4) / 2.0
            k_offset = -(k_max + k_min) / 2.0 / k_scale
            
            normalizer[key] = SingleFieldLinearNormalizer.create_manual(
                scale=1.0 / k_scale, 
                offset=k_offset,
                input_stats_dict={
                    'min': k_min, 'max': k_max,
                    'mean': (k_max + k_min) / 2.0, 'std': k_scale
                }
            )

        # Save to cache
        print(f"Saving normalization stats of {list(normalizer.params_dict.keys())} to {stats_path}")
        torch.save(normalizer.state_dict(), stats_path)
        
        normalizer.to(torch.float32)

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _get_synced_obs_slice(self, raw_indices, zarr_array):
        """
        Extracts a short observation history synced to the trajectory sampler.
        Handles padding automatically if the policy asks for history that doesn't exist 
        (e.g., at the very first step of an episode).
        """

        # 'disk' indices tell us which physical rows to read from the Zarr hard drive array.
        # 'tensor' indices tell us where to insert that data inside our final obs_slice.
        disk_start, disk_end, tensor_insert_start, tensor_insert_end = raw_indices
        
        # Calculating exact disc read limits
        # E.g., if we need 2 observation steps, but 'tensor_insert_start' is 1 (meaning 1 slot 
        # is reserved for padding), we only need max(1, 2 - 1) = 1 real frames from disk
        num_frames_needed = max(1, self.n_obs_steps - tensor_insert_start)
        
        # Failsafe: Never attempt to read past the physical end of the episode data
        num_frames_needed = min(num_frames_needed, disk_end - disk_start) 
        
        # Read strictly what is required
        raw_disk_frames = zarr_array[disk_start : disk_start + num_frames_needed]
        
        # Create the final output tensor
        obs_slice = np.zeros(
            (self.n_obs_steps,) + raw_disk_frames.shape[1:], 
            dtype=raw_disk_frames.dtype
        )
        
        # =====================================================================
        # CASE A: Start-of-episode padding (missing history)
        # Condition: `tensor_insert_start > 0`
        # Meaning: The real data must start later in the tensor because we are at 
        # the beginning of the episode (e.g., t=0) and past history doesn't exist.
        # Action: Duplicate the very first available frame backwards in time.
        # =====================================================================
        if tensor_insert_start > 0:
            pad_count = min(tensor_insert_start, self.n_obs_steps)
            obs_slice[:pad_count] = raw_disk_frames[0]
            
        # =====================================================================
        # CASE B: inserting real data chronologically
        # Condition: `tensor_insert_start < self.n_obs_steps`
        # Meaning: Our observation window is wide enough that at least some 
        # real data belongs in it. (If it were False, the window would be 100% padding).
        # Action: Drop the real frames into their proper chronological slots.
        # =====================================================================
        if tensor_insert_start < self.n_obs_steps:
            real_data_count = min(self.n_obs_steps - tensor_insert_start, num_frames_needed)
            insert_end = tensor_insert_start + real_data_count
            
            obs_slice[tensor_insert_start : insert_end] = raw_disk_frames[:real_data_count]
            
            # =================================================================
            # CASE C: End-of-episode padding (missing future)
            # Condition: `insert_end < self.n_obs_steps`
            # Meaning: Even after inserting all available real data, obs_slice 
            # still isn't full. The episode abruptly ended.
            # Action: Duplicate the very last available frame forwards in time.
            # =================================================================
            if insert_end < self.n_obs_steps:
                obs_slice[insert_end:] = raw_disk_frames[-1]
                
        return obs_slice

    def _sample_to_data(self, sample, skip_subsampling=False):
        """ 
        Returns data as dict of torch tensors. 
        NOTE: 20th dimension is the active Gaussians mask  
        """

        agent_pos = torch.from_numpy(sample['state'])
        action = torch.from_numpy(sample['action'])
        gsplats = torch.from_numpy(sample['gsplats'])

        if not skip_subsampling:
            # Filter out non-active Gaussians at init timestep of sample
            # NOTE: Gaussians becoming active later don't matter as we do not consider them in farthest point sampling
            active_gaussians_mask = gsplats[0, :, 19].to(torch.bool)
            active_gsplats= gsplats[:, active_gaussians_mask, :]

            T, N, D = active_gsplats.shape
        
            if N >= self.INTERMEDIATE_SIZE:
                indices = torch.randperm(N)[:self.INTERMEDIATE_SIZE]
                active_gsplats = active_gsplats[:, indices, :]
                valid_length = self.INTERMEDIATE_SIZE
            else:
                # Zero Padding
                # Safe because 'lengths' will tell the GPU to ignore these zeros during fps sampling
                pad_size = self.INTERMEDIATE_SIZE - N
                padding = torch.zeros((T, pad_size, D), dtype=active_gsplats.dtype)
                active_gsplats = torch.cat([active_gsplats, padding], dim=1)
                valid_length = N
                
            gsplats = active_gsplats
        else:
            # Failsafe if skip_subsampling is used
            valid_length = gsplats.shape[1]

        data = {
            'obs': {
                'gs_positions': gsplats[..., :3],
                'point_cloud': gsplats[..., :3],
                'gs_rotations_9d': gsplats[..., 3:12],
                'gs_surface_normals': gsplats[..., 20:],
                'gs_log_scales': gsplats[..., 12:15],
                'gs_opacities': gsplats[..., 15:16],
                'gs_rgb': gsplats[..., 16:19],
                # CRITICAL: Return valid length for fps sampling during training
                'gs_length': torch.tensor(valid_length, dtype=torch.long), 
                'agent_pos': agent_pos,
            },
            'action': action,
        }
        
        # A sample generated during get_normalizer does not contain actor_keys
        if hasattr(self, 'actor_keys') and len(self.actor_keys) > 0 and all(k in sample for k in self.actor_keys):
            data['actor_poses'] = {k: torch.from_numpy(sample[k]) for k in self.actor_keys}

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # start = time.time()

        sample = self.sampler.sample_sequence(idx)
        raw_indices = self.sampler.indices[idx]
        
        # Get synced obs slices for state and gsplats
        sample['state'] = self._get_synced_obs_slice(raw_indices, self.state_array)
        sample['gsplats'] = self._get_synced_obs_slice(raw_indices, self.gsplats_array)
        
        torch_data = self._sample_to_data(sample)
        
        # print(f"__getitem__ took: {time.time() - start} seconds")
        
        return torch_data
