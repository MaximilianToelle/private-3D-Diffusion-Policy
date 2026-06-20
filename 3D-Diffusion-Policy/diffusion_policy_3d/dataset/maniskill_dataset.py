import os 
import time
from typing import Dict
import torch
import numpy as np
import copy
from tqdm import tqdm
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


class ManiSkillDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.zarr_path = zarr_path
        self.task_name = task_name
        self.n_obs_steps = n_obs_steps

        self.replay_buffer = ReplayBuffer.create_from_path(
            zarr_path, mode='r')
            
        self.actor_keys = [k for k in self.replay_buffer.keys() if k.startswith('actor_pose_')]
        self.keys = ['action', 'state', 'point_cloud'] + self.actor_keys

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

    def get_normalizer(self, mode='limits', **kwargs):
        print(f"Computing normalization stats over the training dataset...")
        
        norm_mask = self.train_mask
        norm_keys = ['action', 'state', 'point_cloud']   # actor poses are only used for reproducing init states
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
            'point_cloud': {'min': None, 'max': None},
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
            # NOTE: we do normalization on farthest point samples!
            # As we only care about min/max it is fine to not use the full scene pcd for normalization
            sample = normalization_sampler.sample_sequence(idx)
            torch_data = self._sample_to_data(sample)   
            obs = torch_data['obs']
            
            # Update min/max bounds
            update_min_max('action', torch_data['action'])
            update_min_max('agent_pos', obs['agent_pos'])
            update_min_max('point_cloud', obs['point_cloud'])

        # 3. Compute final statistics from trackers
        normalizer = LinearNormalizer()

        # --- Positions (Preserving 3D Aspect Ratio) ---
        pos_min = stats['point_cloud']['min']
        pos_max = stats['point_cloud']['max']
        geometric_center = (pos_max + pos_min) / 2.0
        max_radius = torch.clamp((pos_max - pos_min).max() / 2.0, min=1e-4)

        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.ones_like(geometric_center) / max_radius,
            offset=-geometric_center / max_radius,
            input_stats_dict={
                'min': geometric_center - max_radius, 'max': geometric_center + max_radius,
                'mean': geometric_center, 'std': pos_max - pos_min
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

    def _sample_to_data(self, sample):
        # Policy only takes in n_obs_step of the overall sequence of observations
        agent_pos = torch.from_numpy(sample['state'][:self.n_obs_steps])
        point_cloud = torch.from_numpy(sample['point_cloud'][:self.n_obs_steps])
        action = torch.from_numpy(sample['action'])

        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
            },
            'action': action
        }
        
        if hasattr(self, 'actor_keys') and len(self.actor_keys) > 0 and all(k in sample for k in self.actor_keys):
            data['actor_poses'] = {k: torch.from_numpy(sample[k]) for k in self.actor_keys}

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # start = time.time()

        sample = self.sampler.sample_sequence(idx)
        torch_data = self._sample_to_data(sample)
        
        # print(f"__getitem__ took: {time.time() - start} seconds")
        
        return torch_data
