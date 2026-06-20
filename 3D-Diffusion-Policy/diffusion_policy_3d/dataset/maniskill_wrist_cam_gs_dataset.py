import os
import hashlib
from typing import Dict, List
import torch
import numpy as np
import time
import zarr
from tqdm import tqdm
import copy
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

from pytorch3d.ops import sample_farthest_points


def _discover_zarr_paths(zarr_path) -> List[str]:
    """
    Resolve zarr_path into a list of concrete zarr dataset paths.
    
    Three modes:
      1. Single zarr dataset (str with .zgroup inside) -> [zarr_path]
      2. Directory containing multiple zarr subdatasets  -> sorted list of all zarr children
      3. Explicit list of paths                          -> list(zarr_path)
    """
    if not isinstance(zarr_path, str):
        # Mode 3: explicit list (from Hydra ListConfig or Python list)
        paths = list(zarr_path)
        for p in paths:
            assert os.path.isdir(p), f"Zarr path does not exist: {p}"
        return paths
    
    zarr_path = os.path.expanduser(zarr_path)
    
    if os.path.isfile(os.path.join(zarr_path, '.zgroup')):
        # Mode 1: single zarr dataset
        return [zarr_path]
    
    # Mode 2: parent directory containing zarr children
    children = []
    for name in sorted(os.listdir(zarr_path)):
        child = os.path.join(zarr_path, name)
        if os.path.isdir(child) and os.path.isfile(os.path.join(child, '.zgroup')):
            children.append(child)
    
    assert len(children) > 0, (
        f"zarr_path '{zarr_path}' is neither a zarr dataset nor a directory "
        f"containing zarr datasets (no .zgroup files found in children)"
    )
    return children


class WristCamGSManiskillDataset(BaseDataset):
    INTERMEDIATE_SIZE = 32768   # doing further down-sampling on GPU after batch generation
    
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
        self.n_obs_steps = n_obs_steps

        # =====================================================================
        # Resolve zarr_path into concrete dataset paths
        # =====================================================================
        self.zarr_paths = _discover_zarr_paths(zarr_path)

        # =====================================================================
        # Open all replay buffers (lazy zarr reads, no RAM pressure)
        # =====================================================================
        self.replay_buffers: List[ReplayBuffer] = []
        self.gsplats_arrays = []
        self.state_arrays = []
        
        for path in self.zarr_paths:
            buf = ReplayBuffer.create_from_path(path, mode='r')
            self.replay_buffers.append(buf)
            self.gsplats_arrays.append(buf.root['data']['gsplats'])
            self.state_arrays.append(buf.root['data']['state'])

        # =====================================================================
        # Read gs_params / gs_param_sizes from the first buffer, verify all match
        # =====================================================================
        first_buf = self.replay_buffers[0]
        self.actor_keys = [k for k in first_buf.keys() if k.startswith('actor_pose_')]
        self.full_seq_keys = ['action'] + self.actor_keys
        
        meta = first_buf.root['meta']
        attrs = meta if isinstance(meta, dict) else meta.attrs
        
        try:
            self.gs_params = list(attrs['gs_params'])
            self.gs_param_sizes = list(attrs['gs_param_sizes'])
        except KeyError:
            self.gs_params = ["positions", "rotations_9d", "log_scales", "opacities", "rgbs", "active_gaussians_mask", "surf_normals", "semantics"]
            self.gs_param_sizes = [3, 9, 3, 1, 3, 1, 3, 1]
        
        # Safety: verify all buffers have identical gs layout
        for i, buf in enumerate(self.replay_buffers[1:], start=1):
            m = buf.root['meta']
            a = m if isinstance(m, dict) else m.attrs
            try:
                assert list(a['gs_params']) == self.gs_params, \
                    f"gs_params mismatch between dataset 0 and {i}"
                assert list(a['gs_param_sizes']) == self.gs_param_sizes, \
                    f"gs_param_sizes mismatch between dataset 0 and {i}"
            except KeyError:
                pass
            
        self.param_to_obs_key = {
            'positions': 'gs_positions',
            'rotations_9d': 'gs_rotations_9d',
            'log_scales': 'gs_log_scales',
            'opacities': 'gs_opacities',
            'rgbs': 'gs_rgb',
            'surf_normals': 'gs_surface_normals',
            'semantics': 'gs_semantics',
            'active_gaussians_mask': 'gs_active_gaussians_mask'
        }

        # =====================================================================
        # Compute train/val masks GLOBALLY across all buffers
        # =====================================================================
        self.episode_counts = [buf.n_episodes for buf in self.replay_buffers]
        self.total_episodes = sum(self.episode_counts)
        self._episode_cumcounts = np.cumsum(self.episode_counts)
        
        global_val_mask = get_val_mask(
            n_episodes=self.total_episodes,
            val_ratio=val_ratio,
            seed=seed)
            
        global_train_mask = ~global_val_mask
        global_train_mask = downsample_mask(
            mask=global_train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # Slice global masks into per-buffer masks
        self.per_buffer_train_masks = []
        self.per_buffer_val_masks = []
        offset = 0
        for count in self.episode_counts:
            self.per_buffer_train_masks.append(global_train_mask[offset:offset + count])
            self.per_buffer_val_masks.append(global_val_mask[offset:offset + count])
            offset += count

        # =====================================================================
        # Create per-buffer samplers for training
        # =====================================================================
        self.samplers: List[SequenceSampler] = []
        for buf, train_mask in zip(self.replay_buffers, self.per_buffer_train_masks):
            self.samplers.append(SequenceSampler(
                replay_buffer=buf,
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask,
                keys=self.full_seq_keys
            ))

        # Precompute cumulative sample counts for O(log N) global→local index mapping
        self._sampler_lengths = [len(s) for s in self.samplers]
        self._cumulative_lengths = np.cumsum(self._sampler_lengths)
            
        self.global_train_mask = global_train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        n_buf = len(self.replay_buffers)
        print(f"[WristCamGSManiskillDataset] Loaded {n_buf} zarr dataset(s), "
              f"{self.total_episodes} total episodes, "
              f"{sum(self._sampler_lengths)} training samples")

    # =====================================================================
    # Index mapping helpers
    # =====================================================================
    def _global_sample_to_local(self, global_idx):
        """Map a global sample index to (buffer_idx, local_sample_idx)."""
        buf_idx = int(np.searchsorted(self._cumulative_lengths, global_idx, side='right'))
        local_idx = global_idx if buf_idx == 0 else global_idx - int(self._cumulative_lengths[buf_idx - 1])
        return buf_idx, int(local_idx)

    def _global_episode_to_local(self, global_episode_idx):
        """Map a global episode index to (buffer_idx, local_episode_idx)."""
        buf_idx = int(np.searchsorted(self._episode_cumcounts, global_episode_idx, side='right'))
        local_idx = global_episode_idx if buf_idx == 0 else global_episode_idx - int(self._episode_cumcounts[buf_idx - 1])
        return buf_idx, int(local_idx)

    def get_episode_init_data(self, global_episode_idx):
        """
        Return init state data for a specific global episode index.
        Used by the env runner to reproduce initial conditions from the dataset.
        
        Returns:
            init_state: dict with 'actor_poses' and 'agent_pos'
            expert_q_sequence: np.ndarray of the full expert state trajectory
        """
        buf_idx, local_ep_idx = self._global_episode_to_local(global_episode_idx)
        buf = self.replay_buffers[buf_idx]
        
        start_idx = int(buf.episode_ends[local_ep_idx - 1]) if local_ep_idx > 0 else 0
        end_idx = int(buf.episode_ends[local_ep_idx])
        
        init_state = dict()
        if len(self.actor_keys) > 0:
            init_state['actor_poses'] = {
                k: buf[k][start_idx] for k in self.actor_keys
            }
        init_state['agent_pos'] = buf['state'][start_idx]
        
        expert_q_sequence = buf['state'][start_idx:end_idx]
        return init_state, expert_q_sequence

    # =====================================================================
    # Validation dataset
    # =====================================================================
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.samplers = []
        for buf, val_mask in zip(self.replay_buffers, self.per_buffer_val_masks):
            val_set.samplers.append(SequenceSampler(
                replay_buffer=buf,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=val_mask,
                keys=self.full_seq_keys,
            ))
        val_set._sampler_lengths = [len(s) for s in val_set.samplers]
        val_set._cumulative_lengths = np.cumsum(val_set._sampler_lengths)
        val_set.global_train_mask = ~self.global_train_mask
        # Swap masks so get_episode_init_data picks from val episodes
        val_set.per_buffer_train_masks = self.per_buffer_val_masks
        val_set.per_buffer_val_masks = self.per_buffer_train_masks
        return val_set

    # =====================================================================
    # Normalization
    # =====================================================================
    def get_normalizer(self, **kwargs):
        print(f"Computing normalization stats over {len(self.replay_buffers)} training dataset(s)...")
        
        stats = {
            'action': {'min': None, 'max': None},
            'agent_pos': {'min': None, 'max': None},
        }
        if 'positions' in self.gs_params:
            stats['gs_positions'] = {'min': None, 'max': None}
        if 'log_scales' in self.gs_params:
            # For Gaussian normalization, we need mean and std.
            # We track count, sum, and sum of squares (using float64 to prevent numerical instability)
            stats['gs_log_scales'] = {'count': 0, 'sum': 0.0, 'sum_sq': 0.0} 

        def update_min_max(key, tensor):
            flat_tensor = tensor.reshape(-1, tensor.shape[-1])
            batch_min = flat_tensor.min(dim=0)[0]
            batch_max = flat_tensor.max(dim=0)[0]
            if stats[key]['min'] is None:
                stats[key]['min'] = batch_min
                stats[key]['max'] = batch_max
            else:
                stats[key]['min'] = torch.minimum(stats[key]['min'], batch_min)
                stats[key]['max'] = torch.maximum(stats[key]['max'], batch_max)

        # Stream over ALL buffers sequentially
        for buf_i, buf in enumerate(self.replay_buffers):
            norm_mask = self.per_buffer_train_masks[buf_i]
            norm_keys = ['action', 'state', 'gsplats']      # actor poses are only used for reproducing init states
            normalization_sampler = SequenceSampler(
                replay_buffer=buf, 
                sequence_length=1, pad_before=0, pad_after=0,
                episode_mask=norm_mask, keys=norm_keys
            )

            seq_length = getattr(normalization_sampler, 'sequence_length')
            indices = range(len(normalization_sampler))[::seq_length]
            
            desc = f"Streaming dataset {buf_i+1}/{len(self.replay_buffers)} for normalization stats"
            for idx in tqdm(indices, desc=desc):
                sample = normalization_sampler.sample_sequence(idx)
                torch_data = self._sample_to_data(sample, skip_subsampling=True)    # IMPORTANT to see full GS scene during normalization
                obs = torch_data['obs']
                
                update_min_max('action', torch_data['action'])
                update_min_max('agent_pos', obs['agent_pos'])
                if 'positions' in self.gs_params:
                    update_min_max('gs_positions', obs['gs_positions'])
                
                if 'log_scales' in self.gs_params:
                # Update sum and sum of squares for log_scales
                # NOTE: We compute a single, global mean and std across scaling dimensions
                # -> A Gaussian can rotate 90 degree and swap its x- and y-scales without changing its appearance!   
                # NOTE: We cast to .double() to prevent catastrophic cancellation in large sum operations
                    log_scales_flat = obs['gs_log_scales'].reshape(-1).double()
                    stats['gs_log_scales']['count'] += log_scales_flat.shape[0]
                    stats['gs_log_scales']['sum'] += log_scales_flat.sum(dim=0)
                    stats['gs_log_scales']['sum_sq'] += (log_scales_flat ** 2).sum(dim=0)

        # Build normalizer from accumulated stats
        normalizer = LinearNormalizer()

        # --- Positions (Preserving 3D Aspect Ratio) ---
        if 'positions' in self.gs_params:
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
        if 'log_scales' in self.gs_params:
            N = stats['gs_log_scales']['count']
            mean_log_scales = stats['gs_log_scales']['sum'] / N
            var_log_scales = stats['gs_log_scales']['sum_sq'] / N - (mean_log_scales ** 2)
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
        if 'rotations_9d' in self.gs_params:
            normalizer['gs_rotations_9d'] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)
        if 'surf_normals' in self.gs_params:
            normalizer['gs_surface_normals'] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)

        if 'opacities' in self.gs_params:
            # got processed with sigmoid -> [0, 1]
            normalizer['gs_opacities'] = SingleFieldLinearNormalizer.create_manual(
                scale=torch.tensor([2.0], dtype=torch.float32), 
                offset=torch.tensor([-1.0], dtype=torch.float32),
                input_stats_dict={
                    'min': torch.tensor([0.0]), 'max': torch.tensor([1.0]),
                    'mean': torch.tensor([0.5]), 'std': torch.tensor([0.5])
                }
            )

        if 'rgbs' in self.gs_params:
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
                scale=1.0 / k_scale, offset=k_offset,
                input_stats_dict={
                    'min': k_min, 'max': k_max,
                    'mean': (k_max + k_min) / 2.0, 'std': k_scale
                }
            )

        normalizer.to(torch.float32)
        return normalizer

    def __len__(self) -> int:
        return int(self._cumulative_lengths[-1]) if len(self._cumulative_lengths) > 0 else 0

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
        
        raw_disk_frames = zarr_array[disk_start : disk_start + num_frames_needed]
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
        """
        agent_pos = torch.from_numpy(sample['state'])
        action = torch.from_numpy(sample['action'])
        gsplats = torch.from_numpy(sample['gsplats'])
        D = gsplats.shape[-1]

        if not skip_subsampling:
            # Filter out non-active Gaussians at init timestep of sample
            # NOTE: Gaussians becoming active later don't matter as we do not consider them during sampling
            if 'active_gaussians_mask' in self.gs_params:
                active_idx = sum(self.gs_param_sizes[:self.gs_params.index('active_gaussians_mask')])
                active_gaussians_mask = gsplats[0, :, active_idx].to(torch.bool)
                active_gsplats = gsplats[:, active_gaussians_mask, :]
            else:
                active_gsplats = gsplats

            T, N, D = active_gsplats.shape
        
            if N >= self.INTERMEDIATE_SIZE:
                indices = torch.randperm(N)[:self.INTERMEDIATE_SIZE]
                active_gsplats = active_gsplats[:, indices, :]
                valid_length = self.INTERMEDIATE_SIZE
            else:
                # Zero Padding
                # Safe because 'lengths' will tell the GPU to ignore these zeros during sampling
                pad_size = self.INTERMEDIATE_SIZE - N
                padding = torch.zeros((T, pad_size, D), dtype=active_gsplats.dtype)
                active_gsplats = torch.cat([active_gsplats, padding], dim=1)
                valid_length = N
                
            gsplats = active_gsplats
        else:
            valid_length = gsplats.shape[1]

        obs_dict = {
            # CRITICAL: Return valid length for sampling during training
            'gs_length': torch.tensor(valid_length, dtype=torch.long), 
            'agent_pos': agent_pos,
        }
        
        # Dynamically populate obs_dict based on gs_params layout
        curr = 0
        for p, size in zip(self.gs_params, self.gs_param_sizes):
            obs_key = self.param_to_obs_key.get(p, f"gs_{p}")
            obs_dict[obs_key] = gsplats[..., curr:curr + size]
            curr += size
            
        if 'positions' in self.gs_params:
            obs_dict['point_cloud'] = obs_dict[self.param_to_obs_key['positions']]

        data = {
            'obs': obs_dict,
            'action': action,
        }
        
        # A sample generated during get_normalizer does not contain actor_keys
        if hasattr(self, 'actor_keys') and len(self.actor_keys) > 0 and all(k in sample for k in self.actor_keys):
            data['actor_poses'] = {k: torch.from_numpy(sample[k]) for k in self.actor_keys}

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        buf_idx, local_idx = self._global_sample_to_local(idx)

        sample = self.samplers[buf_idx].sample_sequence(local_idx)
        raw_indices = self.samplers[buf_idx].indices[local_idx]
        
        sample['state'] = self._get_synced_obs_slice(raw_indices, self.state_arrays[buf_idx])
        sample['gsplats'] = self._get_synced_obs_slice(raw_indices, self.gsplats_arrays[buf_idx])
        
        return self._sample_to_data(sample)
