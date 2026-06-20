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


class GSManiskillDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        num_gaussians=1024
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.num_gaussians = num_gaussians

        self.replay_buffer = ReplayBuffer.create_from_path(
            zarr_path, mode='r')
            
        self.actor_keys = [k for k in self.replay_buffer.keys() if k.startswith('actor_pose_')]
        self.keys = ['action', 'state', 'gsplats'] + self.actor_keys
            
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed
        )
            
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
            keys=self.keys
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
            keys=self.keys,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs):
        print(f"Computing normalization stats over the training dataset...")
        
        norm_mask = self.train_mask
        norm_keys = ['action', 'state', 'gsplats']   # actor poses do not get normalized -> only used for reproducing init states
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
            torch_data = self._sample_to_data(sample, skip_random_sampling=True)    # IMPORTANT to see full GS scene during normalization
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

        normalizer.to(torch.float32)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample, skip_random_sampling=False):
        """ 
        Returns data as dict of torch tensors. 
        Dataset has already been processed equally to gs_maniskill_wrapper.py.
        NOTE: We do high-opacity subsampling here on cpu and farthest point sampling within train.py on GPU!
        """

        # Reading the 19D vectors from the zarr sample
        agent_pos = torch.from_numpy(sample['state']).to(torch.float32)
        action = torch.from_numpy(sample['action']).to(torch.float32)
        gsplats = torch.from_numpy(sample['gsplats']).to(torch.float32)

        # agent_pos = torch.from_numpy(sample['state']).to(torch.float32)
        # positions = torch.from_numpy(sample['gs_positions']).to(torch.float32)
        # rotations_9d = torch.from_numpy(sample['gs_rotations_9d']).to(torch.float32)
        # log_scales = torch.from_numpy(sample['gs_log_scales']).to(torch.float32)
        # opacities = torch.from_numpy(sample['gs_opacities']).to(torch.float32)
        # rgb = torch.from_numpy(sample['gs_rgb']).to(torch.float32)
        # action = torch.from_numpy(sample['action']).to(torch.float32)

        if not skip_random_sampling:
            # Performing random sampling on the first timestep
            # sample_farthest_points expects (B, N, 3)
            N_total = gsplats.shape[1]
            sampled_indices = torch.randperm(N_total)[:self.num_gaussians]

            # Subsample the full parameter set as a single copy
            gsplats = gsplats[:, sampled_indices, :] # (T, K, 19)

        positions = gsplats[:, :, :3]
        rotations_9d = gsplats[:, :, 3:12]
        log_scales = gsplats[:, :, 12:15]
        opacities = gsplats[:, :, 15:16]
        rgb = gsplats[:, :, 16:19]

        data = {
            'obs': {
                'gs_positions': positions,
                'gs_rotations_9d': rotations_9d,
                'gs_log_scales': log_scales,
                'gs_opacities': opacities,
                'gs_rgb': rgb,
                'agent_pos': agent_pos,
            },
            'action': action,
        }
        
        # We do not need to normalize actor_keys, so they are not included in such a sample
        if hasattr(self, 'actor_keys') and len(self.actor_keys) > 0 and all(k in sample for k in self.actor_keys):
            data['actor_poses'] = {k: torch.from_numpy(sample[k]).to(torch.float32) for k in self.actor_keys}

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # start = time.time()
        sample = self.sampler.sample_sequence(idx)
        torch_data = self._sample_to_data(sample)
        # print(f"__getitem__ took: {time.time() - start} seconds")
        
        return torch_data
