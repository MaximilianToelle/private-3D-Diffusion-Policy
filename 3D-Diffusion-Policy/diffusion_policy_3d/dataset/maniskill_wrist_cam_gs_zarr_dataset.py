import os
import hashlib
from typing import Dict, List, Iterator
import torch
from torch.utils.data import Sampler
import numpy as np
import time
import zarr
from tqdm import tqdm
import copy
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, PerTimestepLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.dataset_utils import _pose7_to_mat_np, _mat_to_pose9d_np

from pytorch3d.ops import sample_farthest_points


# TODO: Since we are back to loading all datasets into RAM, we could further optimize this code! 


# class DatasetBlockSampler(Sampler[int]):
#     """
#     Treats each Zarr dataset as a single block. 
#     Shuffles the order of the datasets, and shuffles the samples within each dataset.
#     Exhausts one dataset completely before moving to the next.
#     """
#     def __init__(self, cumulative_lengths: List[int]):
#         self.cumulative_lengths = cumulative_lengths
#         self.num_datasets = len(cumulative_lengths)
        
#         # Each block is simply an entire dataset
#         self.dataset_blocks = []
#         start_idx = 0
#         for end_idx in cumulative_lengths:
#             self.dataset_blocks.append(np.arange(start_idx, end_idx))
#             start_idx = end_idx
    
#     def __iter__(self) -> Iterator[int]:
#         # 1. Shuffle the order of the datasets (e.g., [Dataset 2, Dataset 0, Dataset 1])
#         dataset_order = np.random.permutation(self.num_datasets)
        
#         for ds_idx in dataset_order:
#             # 2. Fully shuffle the indices inside this specific dataset
#             ds_indices = self.dataset_blocks[ds_idx]
#             shuffled_indices = np.random.permutation(ds_indices)
            
#             # 3. Yield single integers (PyTorch DataLoader will group them into batches of 128!)
#             for idx in shuffled_indices:
#                 yield int(idx)
    
#     def __len__(self) -> int:
#         return int(self.cumulative_lengths[-1]) if len(self.cumulative_lengths) > 0 else 0


class WristCamGSManiskillDataset(BaseDataset):
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
        representation_space="abs_joint_pos",  # abs_joint_pos | relative_ee_pose
        verbose=False,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.verbose = verbose
        assert representation_space in ("abs_joint_pos", "relative_ee_pose"), \
            f"representation_space must be 'abs_joint_pos' or 'relative_ee_pose', got '{representation_space}'"
        self.representation_space = representation_space

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
            # buf = ReplayBuffer.create_from_path(path, mode='r')
            buf = ReplayBuffer.copy_from_path(path)
            
            self.replay_buffers.append(buf)
            self.gsplats_arrays.append(buf.root['data']['gsplats'])
            self.state_arrays.append(buf.root['data']['state'])
 
        # =====================================================================
        # Read gs_params / gs_param_sizes from the first buffer, verify all match
        # =====================================================================
        first_buf = self.replay_buffers[0]
        self.actor_keys = [k for k in first_buf.keys() if k.startswith('actor_pose_')]
        if self.representation_space == "abs_joint_pos":
            self.full_seq_keys = ['action'] + self.actor_keys
        elif self.representation_space == "relative_ee_pose":
            # Need action (for gripper dim) and tcp_pose (for relative EE computation) over the full horizon
            self.full_seq_keys = ['action', 'tcp_pose'] + self.actor_keys

        meta = first_buf.root['meta']
        attrs = meta if isinstance(meta, dict) else meta.attrs
        
        try:
            self.gs_params = list(attrs['gs_params'])
            self.gs_param_sizes = list(attrs['gs_param_sizes'])
        except KeyError:
            self.gs_params = ["positions", "rotations_9d", "log_scales", "opacities", "rgbs", "active_gaussians_mask", "surf_normals", "semantics"]
            self.gs_param_sizes = [3, 9, 3, 1, 3, 1, 3, 1]
        
        # Static mapping for parsing the flattened gaussian data
        self.param_slices = {}
        curr = 0
        for p, size in zip(self.gs_params, self.gs_param_sizes):
            self.param_slices[p] = slice(curr, curr + size)
            curr += size

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
        
        if self.representation_space == "abs_joint_pos":
            return self._get_normalizer_abs_joint_pos()
        elif self.representation_space == "relative_ee_pose":
            return self._get_normalizer_relative_ee_pose()

    # -----------------------------------------------------------------
    # Normalizer: absolute joint position mode (existing min/max logic)
    # -----------------------------------------------------------------
    def _get_normalizer_abs_joint_pos(self):
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
                torch_data = self._sample_to_data(sample)
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

    # -----------------------------------------------------------------
    # Normalizer: relative EE pose mode (TRI-LBM clamped percentile normalization)
    # -----------------------------------------------------------------
    def _get_normalizer_relative_ee_pose(self):
        """
        TRI-LBM-Style per-timestep, per-feature-dim percentile normalization:
            yi = clamp(2 * (xi - x^0.02) / (x^0.98 - x^0.02) - 1,  -1.5, 1.5)
        
        Percentiles are computed per (timestep, feature_dim) independently.
        
        To avoid OOM when accumulating all samples across datasets, we process
        one dataset at a time: for each dataset we collect all values per (t, d),
        flatten and sort them into compact 1D tensors. After all datasets are 
        processed, we compute percentiles across datasets via binary search over
        the sorted chunks using torch.searchsorted.
        """
        # We need full-horizon sequences (same as training) to get per-timestep stats
        T_obs = self.n_obs_steps
        T_horizon = self.horizon

        # Probe feature dimensions from first sample
        probe_sample = self.samplers[0].sample_sequence(0)
        probe_sample['state'] = self._get_synced_obs_slice(
            self.samplers[0].indices[0], self.state_arrays[0], name="state")
        probe_sample['gsplats'] = self._get_synced_obs_slice(
            self.samplers[0].indices[0], self.gsplats_arrays[0], name="gsplats")
        probe_data = self._sample_to_data(probe_sample)
        D_agent_pos = probe_data['obs']['agent_pos'].shape[-1]
        D_action = probe_data['action'].shape[-1]
        D_gs_positions = 3

        # =====================================================================
        # Phase 1: Collect and sort data per dataset
        # =====================================================================
        # For each (timestep, feature_dim), we store a list of sorted 1D tensors,
        # one per dataset. This avoids holding all raw samples in memory at once.
        # Layout: sorted_chunks[t][d] = [sorted_1d_tensor_from_ds0, sorted_1d_tensor_from_ds1, ...]
        agent_pos_sorted_chunks = [[[] for _ in range(D_agent_pos)] for _ in range(T_obs)]
        action_sorted_chunks = [[[] for _ in range(D_action)] for _ in range(T_horizon)]
        gs_positions_sorted_chunks = [[[] for _ in range(D_gs_positions)] for _ in range(T_obs)]

        for buf_i, buf in enumerate(self.replay_buffers):
            norm_mask = self.per_buffer_train_masks[buf_i]
            normalization_sampler = SequenceSampler(
                replay_buffer=buf,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=norm_mask,
                keys=self.full_seq_keys,  # sampler only handles these keys
            )

            # Accumulate raw values for this single dataset
            dataset_agent_pos_vals = []
            dataset_action_vals = []
            dataset_gs_positions_vals = []
            desc = f"Collecting stats from dataset {buf_i+1}/{len(self.replay_buffers)}"
            for idx in tqdm(range(len(normalization_sampler)), desc=desc):
                sample = normalization_sampler.sample_sequence(idx)
                raw_indices = normalization_sampler.indices[idx]
                sample['state'] = self._get_synced_obs_slice(
                    raw_indices, self.state_arrays[buf_i], name="state")
                sample['gsplats'] = self._get_synced_obs_slice(
                    raw_indices, self.gsplats_arrays[buf_i], name="gsplats")

                torch_data = self._sample_to_data(sample)

                dataset_agent_pos_vals.append(torch_data['obs']['agent_pos'])        # (T_obs, D_agent_pos)
                dataset_action_vals.append(torch_data['action'])                     # (T_horizon, D_action)
                dataset_gs_positions_vals.append(torch_data['obs']['gs_positions'])  # (T_obs, N_gaussians, 3)

            if len(dataset_agent_pos_vals) == 0:
                continue

            # Stack this dataset's samples, extract per (t, d), sort, and store
            # the compact sorted 1D tensors. Then delete the raw 
            # tensors to free memory before processing the next dataset.
            agent_pos_stacked = torch.stack(dataset_agent_pos_vals, dim=0)  # (N_samples, T_obs, D_agent_pos)
            for t in range(T_obs):
                for d in range(D_agent_pos):
                    agent_pos_sorted_chunks[t][d].append(
                        agent_pos_stacked[:, t, d].contiguous().sort()[0])
            del dataset_agent_pos_vals, agent_pos_stacked

            action_stacked = torch.stack(dataset_action_vals, dim=0)  # (N_samples, T_horizon, D_action)
            for t in range(T_horizon):
                for d in range(D_action):
                    action_sorted_chunks[t][d].append(
                        action_stacked[:, t, d].contiguous().sort()[0])
            del dataset_action_vals, action_stacked

            # gs_positions: each sample has (T_obs, N_gaussians, 3) with fixed N_gaussians.
            # We stack along a new sample dimension, then flatten over both samples and
            # Gaussians to get all values for a given (t, d).
            gs_positions_stacked = torch.stack(dataset_gs_positions_vals, dim=0)  # (N_samples, T_obs, N_gaussians, 3)
            for t in range(T_obs):
                for d in range(D_gs_positions):
                    gs_positions_sorted_chunks[t][d].append(
                        gs_positions_stacked[:, t, :, d].flatten().sort()[0])
            del dataset_gs_positions_vals, gs_positions_stacked

        # =====================================================================
        # Phase 2: Compute percentile over datasets via binary search
        # =====================================================================
        def percentile_over_datasets(sorted_arrays: list, percentile: float) -> float:
            """
            Find the given percentile (0-100) across multiple pre-sorted 1D tensors
            using binary search + torch.searchsorted.
            
            For each candidate value, searchsorted counts how many elements in each
            pre-sorted array are <= that value. We bisect until the count matches the
            target rank.
            
            Convergence: each iteration halves the search interval. After 60 
            iterations the interval width is (hi-lo)/2^60 ~ 1e-16, well below
            float32 machine epsilon (~6e-8). Even 30 iterations would suffice 
            for float32 data, but 60 is essentially free.
            
            Note: unlike torch.quantile, this does NOT interpolate between adjacent 
            data points — it converges to a value in the continuous interval. The 
            difference is below float32 representability, which is negligible for
            computing normalization scale/offset, but means results are not 
            bit-identical to torch.quantile.
            """
            total_elements = sum(arr.numel() for arr in sorted_arrays)
            if total_elements == 0:
                return 0.0

            target_rank = int((percentile / 100.0) * (total_elements - 1))

            lower_bound = min(arr[0].item() for arr in sorted_arrays if arr.numel() > 0)
            upper_bound = max(arr[-1].item() for arr in sorted_arrays if arr.numel() > 0)
            if lower_bound == upper_bound:
                return float(lower_bound)

            # 60 iterations of bisection converges to full float32 precision
            for _ in range(60):
                midpoint = (lower_bound + upper_bound) / 2.0
                midpoint_tensor = torch.tensor(midpoint, dtype=sorted_arrays[0].dtype)
                # Count how many elements across all arrays are <= midpoint
                count_less_or_equal = sum(
                    torch.searchsorted(arr, midpoint_tensor, right=True).item()
                    for arr in sorted_arrays
                )
                if count_less_or_equal <= target_rank:
                    lower_bound = midpoint
                else:
                    upper_bound = midpoint

            return (lower_bound + upper_bound) / 2.0

        def build_percentile_tensors(sorted_chunks, n_timesteps, n_features, lower_p, upper_p):
            """Build (T, D) tensors for lower_p and upper_p from the sorted chunks data structure."""
            p_low = torch.zeros(n_timesteps, n_features)
            p_high = torch.zeros(n_timesteps, n_features)
            for t in range(n_timesteps):
                for d in range(n_features):
                    p_low[t, d] = percentile_over_datasets(sorted_chunks[t][d], lower_p)
                    p_high[t, d] = percentile_over_datasets(sorted_chunks[t][d], upper_p)
            return p_low, p_high

        normalizer = LinearNormalizer()
        lower_percentile = 2
        upper_percentile = 98

        # --- agent_pos ---
        p02_agent_pos, p98_agent_pos = build_percentile_tensors(
            agent_pos_sorted_chunks, T_obs, D_agent_pos, lower_percentile, upper_percentile)
        normalizer['agent_pos'] = PerTimestepLinearNormalizer.create_clamped_percentile_normalizer(
            p02=p02_agent_pos, p98=p98_agent_pos,
            n_timesteps=T_obs, n_features=D_agent_pos,
        )
        del agent_pos_sorted_chunks

        # --- action ---
        p02_action, p98_action = build_percentile_tensors(
            action_sorted_chunks, T_horizon, D_action, lower_percentile, upper_percentile)
        normalizer['action'] = PerTimestepLinearNormalizer.create_clamped_percentile_normalizer(
            p02=p02_action, p98=p98_action,
            n_timesteps=T_horizon, n_features=D_action,
        )
        del action_sorted_chunks

        # --- gs_positions ---
        p02_gs_positions, p98_gs_positions = build_percentile_tensors(
            gs_positions_sorted_chunks, T_obs, D_gs_positions, lower_percentile, upper_percentile)
        normalizer['gs_positions'] = PerTimestepLinearNormalizer.create_clamped_percentile_normalizer(
            p02=p02_gs_positions, p98=p98_gs_positions,
            n_timesteps=T_obs, n_features=D_gs_positions,
        )
        normalizer['point_cloud'] = normalizer['gs_positions']
        del gs_positions_sorted_chunks

        # --- Hardcoded / identity GS normalizers ---
        # Surface normals: already in [-1, 1] by construction (unit vectors).
        # Same reasoning as 6D rotation: avoid corrupting the geometric structure.
        if 'surf_normals' in self.gs_params:
            normalizer['gs_surface_normals'] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)

        if 'rotations_9d' in self.gs_params:
            normalizer['gs_rotations_9d'] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)

        if 'log_scales' in self.gs_params:
            raise NotImplementedError(
                "log_scales normalization is not yet implemented for relative_ee_pose mode. "
                "Consider whether log_scales should also be represented as relative transforms, "
                "since a 4x4 transformation matrix can encode uniform scaling. "
                "For now, exclude 'log_scales' from gs_params when using relative_ee_pose."
            )

        if 'opacities' in self.gs_params:
            normalizer['gs_opacities'] = SingleFieldLinearNormalizer.create_manual(
                scale=torch.tensor([2.0], dtype=torch.float32), 
                offset=torch.tensor([-1.0], dtype=torch.float32),
                input_stats_dict={
                    'min': torch.tensor([0.0]), 'max': torch.tensor([1.0]),
                    'mean': torch.tensor([0.5]), 'std': torch.tensor([0.5])
                }
            )

        if 'rgbs' in self.gs_params:
            normalizer['gs_rgb'] = SingleFieldLinearNormalizer.create_manual(
                scale=torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32), 
                offset=torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32),
                input_stats_dict={
                    'min': torch.zeros(3), 'max': torch.ones(3),
                    'mean': torch.full((3,), 0.5), 'std': torch.full((3,), 0.5)
                }
            )

        normalizer.to(torch.float32)
        return normalizer

    def __len__(self) -> int:
        return int(self._cumulative_lengths[-1]) if len(self._cumulative_lengths) > 0 else 0

    def _get_synced_obs_slice(self, raw_indices, zarr_array, name=""):
        """
        Extracts a short observation history synced to the trajectory sampler.
        Handles padding automatically if the policy asks for history that doesn't exist 
        (e.g., at the very first step of an episode).
        """
        t_start = time.time()

        # 'disk' indices tell us which physical rows to read from the Zarr hard drive array.
        # 'tensor' indices tell us where to insert that data inside our final obs_slice.
        disk_start, disk_end, tensor_insert_start, tensor_insert_end = raw_indices
        
        # Calculating exact disc read limits
        # E.g., if we need 2 observation steps, but 'tensor_insert_start' is 1 (meaning 1 slot 
        # is reserved for padding), we only need max(1, 2 - 1) = 1 real frames from disk
        num_frames_needed = max(1, self.n_obs_steps - tensor_insert_start)
        
        # Failsafe: Never attempt to read past the physical end of the episode data
        num_frames_needed = min(num_frames_needed, disk_end - disk_start) 
        
        t_before_read = time.time()
        raw_disk_frames = zarr_array[disk_start : disk_start + num_frames_needed]
        t_after_read = time.time()
        
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
                
        t_end = time.time()
        if self.verbose:
            pid = os.getpid()
            print(
                f"    [Worker {pid}] _get_synced_obs_slice ({name}):\n"
                f"      - Index extraction: {(t_before_read - t_start) * 1000:.3f} ms\n"
                f"      - Zarr Disk Read:   {(t_after_read - t_before_read) * 1000:.3f} ms\n"
                f"      - Copy & Padding:   {(t_end - t_after_read) * 1000:.3f} ms\n"
                f"      - Total:            {(t_end - t_start) * 1000:.3f} ms"
            )
        return obs_slice

    def _sample_to_data(self, sample):
        """ 
        Returns data as dict of torch tensors.
        Handles two representation modes:
          - abs_joint_pos: pass-through (joint positions as proprio and actions)
          - relative_ee_pose: SE(3) transform GS into anchor frame, relative proprio and actions
        """
        gsplats = sample['gsplats']   # (n_obs_steps, N, D) numpy float16

        if self.representation_space == "abs_joint_pos":
            # =====================================================
            # Absolute joint position mode
            # =====================================================
            agent_pos = torch.from_numpy(sample['state'])
            action = torch.from_numpy(sample['action'])
            
            obs_dict = {'agent_pos': agent_pos}
            
            # Dynamically populate obs_dict based on gs_params layout
            gsplats_torch = torch.from_numpy(gsplats)
            for p, size in zip(self.gs_params, self.gs_param_sizes):
                obs_key = self.param_to_obs_key.get(p, f"gs_{p}")
                s = self.param_slices[p]
                obs_dict[obs_key] = gsplats_torch[..., s]
                
            if 'positions' in self.gs_params:
                obs_dict['point_cloud'] = obs_dict[self.param_to_obs_key['positions']]

            data = {'obs': obs_dict, 'action': action}

        elif self.representation_space == "relative_ee_pose":
            # =====================================================
            # Relative EE pose mode
            # All quantities expressed relative to the anchor ("current time") frame
            # =====================================================
            
            tcp_pose_horizon = sample['tcp_pose'].astype(np.float32)     # (horizon, 7)
            tcp_pose_obs = tcp_pose_horizon[:self.n_obs_steps]           # (n_obs_steps, 7)
            state = sample['state'].astype(np.float32)                   # (n_obs_steps, 9)
            action_raw = sample['action'].astype(np.float32)             # (horizon, 8)
            gsplats = gsplats.astype(np.float32)                         # (n_obs_steps, N, D)

            # --- Anchor frame: TCP at the last observation step ---
            anchor_tcp_to_base = _pose7_to_mat_np(tcp_pose_obs[-1])  # (4, 4)
            anchor_base_to_tcp = np.linalg.inv(anchor_tcp_to_base)           # (4, 4)

            # --- 1. Transform GS observations into the anchor frame ---
            # GS at each obs step i is already in EE(i) frame (from data collection).
            # To express GS(i) in the anchor frame: T_rel = T_anchor^{-1} @ T_ee(i)
            obs_tcp_to_base = _pose7_to_mat_np(tcp_pose_obs)       # (n_obs_steps, 4, 4)
            T_rel_obs = anchor_base_to_tcp[None, :, :] @ obs_tcp_to_base   # (n_obs_steps, 4, 4)
            R_rel_obs = T_rel_obs[:, :3, :3]                # (n_obs_steps, 3, 3)
            t_rel_obs = T_rel_obs[:, :3, 3]                 # (n_obs_steps, 3)

            # Transform positions: p_anchor = R_rel @ p_ee(i) + t_rel
            if 'positions' in self.param_slices:
                s = self.param_slices['positions']
                pos = gsplats[:, :, s]                            # (T_obs, N, 3)
                R_T = R_rel_obs.transpose(0, 2, 1)                # (T_obs, 3, 3)
                # Vectorized: (T_obs, N, 3) @ (T_obs, 3, 3) + (T_obs, 1, 3)
                gsplats[:, :, s] = (pos @ R_T) + t_rel_obs[:, None, :]

            # Transform 9D rotations: R_anchor = R_rel @ R_ee(i)
            if 'rotations_9d' in self.param_slices:
                # TODO: Better provide 6D rotations
                s = self.param_slices['rotations_9d']
                rot9d = gsplats[:, :, s]                          # (T_obs, N, 9)
                T_obs, N = rot9d.shape[:2]
                rot_mats = rot9d.reshape(T_obs, N, 3, 3)         # (T_obs, N, 3, 3)
                # (T_obs, 1, 3, 3) @ (T_obs, N, 3, 3)
                rot_mats = R_rel_obs[:, None, :, :] @ rot_mats
                gsplats[:, :, s] = rot_mats.reshape(T_obs, N, 9)

            # Transform surface normals: n_anchor = R_rel @ n_ee(i)
            if 'surf_normals' in self.param_slices:
                s = self.param_slices['surf_normals']
                normals = gsplats[:, :, s]                        # (T_obs, N, 3)
                R_T = R_rel_obs.transpose(0, 2, 1)                # (T_obs, 3, 3)
                gsplats[:, :, s] = normals @ R_T

            # Populate obs_dict from transformed gsplats
            gsplats_torch = torch.from_numpy(gsplats)
            obs_dict = {}
            for p, size in zip(self.gs_params, self.gs_param_sizes):
                obs_key = self.param_to_obs_key.get(p, f"gs_{p}")
                s = self.param_slices[p]
                obs_dict[obs_key] = gsplats_torch[..., s]

            if 'positions' in self.gs_params:
                obs_dict['point_cloud'] = obs_dict[self.param_to_obs_key['positions']]

            # --- 2. Proprioception: relative EE pose + gripper ---
            # pos(3) + rot6d(6) + gripper(2) = 11D per obs step
            rel_obs_pose9d = _mat_to_pose9d_np(T_rel_obs)  # (n_obs_steps, 9)
            # TODO: If we represent the gripper action in 1D, then a 1D state should also be fine? 
            gripper_state = state[:, 7:9]                   # (n_obs_steps, 2)
            # NOTE: using the absolute gripper state
            agent_pos_np = np.concatenate([rel_obs_pose9d, gripper_state], axis=-1)  # (n_obs_steps, 11)
            obs_dict['agent_pos'] = torch.from_numpy(agent_pos_np)

            # --- 3. Actions: relative EE pose + gripper ---
            # pos(3) + rot6d(6) + gripper(1) = 10D per action step
            full_horizon_tcp_to_base = _pose7_to_mat_np(tcp_pose_horizon)   # (horizon, 4, 4)
            T_rel_action = anchor_base_to_tcp[None, :, :] @ full_horizon_tcp_to_base # (horizon, 4, 4)
            rel_action_pose9d = _mat_to_pose9d_np(T_rel_action)  # (horizon, 9)
            # NOTE: using the absolute gripper action
            gripper_action = action_raw[:, -1:]                   # (horizon, 1)
            action_np = np.concatenate([rel_action_pose9d, gripper_action], axis=-1)  # (horizon, 10)
            action = torch.from_numpy(action_np)

            data = {'obs': obs_dict, 'action': action}

        # A sample generated during get_normalizer does not contain actor_keys
        if hasattr(self, 'actor_keys') and len(self.actor_keys) > 0 and all(k in sample for k in self.actor_keys):
            data['actor_poses'] = {k: torch.from_numpy(sample[k]) for k in self.actor_keys}

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t_1 = time.time()

        buf_idx, local_idx = self._global_sample_to_local(idx)
        t_2 = time.time()

        # Track dataset block switches
        # if not hasattr(self, "_worker_current_buf"):
        #     self._worker_current_buf = {}
        # pid = os.getpid()
        # last_buf = self._worker_current_buf.get(pid, -1)
        # if buf_idx != last_buf:
        #     print(f"[Worker {pid}] Switched to reading from dataset {buf_idx} (prev: {last_buf})")
        #     self._worker_current_buf[pid] = buf_idx
        # t_3 = time.time()
        t_3 = t_2
        
        sample = self.samplers[buf_idx].sample_sequence(local_idx)
        t_4 = time.time()
        
        raw_indices = self.samplers[buf_idx].indices[local_idx]
        t_5 = time.time()
        
        sample['state'] = self._get_synced_obs_slice(raw_indices, self.state_arrays[buf_idx], name="state")
        sample['gsplats'] = self._get_synced_obs_slice(raw_indices, self.gsplats_arrays[buf_idx], name="gsplats")
        t_6 = time.time()

        data = self._sample_to_data(sample)
        t_7 = time.time()

        if self.verbose:
            pid = os.getpid()
            print(
                f"\n[Worker {pid}] __getitem__ timings (idx: {idx}, buf: {buf_idx}, local: {local_idx}):\n"
                f"  - Index Mapping:         {(t_2 - t_1) * 1000:.3f} ms\n"
                f"  - Block Switch Tracking: {(t_3 - t_2) * 1000:.3f} ms\n"
                f"  - Sequence Sampling:     {(t_4 - t_3) * 1000:.3f} ms\n"
                f"  - Retrieve Indices:      {(t_5 - t_4) * 1000:.3f} ms\n"
                f"  - Load Zarr Obs Slices:  {(t_6 - t_5) * 1000:.3f} ms\n"
                f"  - Sample to Torch Data:  {(t_7 - t_6) * 1000:.3f} ms\n"
                f"  - Total __getitem__:     {(t_7 - t_1) * 1000:.3f} ms"
            )

        return data


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


if __name__ == "__main__":
    import hydra
    from torch.utils.data import DataLoader
    import time
    import tqdm
    from diffusion_policy_3d.common.pytorch_util import dict_apply
    from diffusion_policy_3d.dataset.data_augmentations import GaussianCompose
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    @hydra.main(
        version_base=None,
        config_path="../config",
        config_name="wrist_cam_gsplat_dp3"
    )
    def main(cfg):
        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseDataset), f"dataset must be BaseDataset, got {type(dataset)}"
        print(f"Dataset instantiated. Length: {len(dataset)}")
        
        # dataset_block_sampler = DatasetBlockSampler(dataset._cumulative_lengths)
        # train_dataloader = DataLoader(dataset, sampler=dataset_block_sampler, **cfg.dataloader)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        print("DataLoader instantiated.")

        # =========================================================
        # Test Normalization
        # =========================================================
        print(f"\n--- Testing Normalizer ({dataset.representation_space}) ---")
        normalizer = dataset.get_normalizer()
        
        # Grab one batch to test normalization
        batch = next(iter(train_dataloader))
        print("Got test batch. Normalizing...")
        
        # The normalizer expects a dict with keys matching its params_dict
        test_dict = {
            'action': batch['action'],
            'agent_pos': batch['obs']['agent_pos'],
        }
        if 'positions' in dataset.gs_params:
            test_dict['gs_positions'] = batch['obs']['gs_positions']
            
        normalized_dict = normalizer.normalize(test_dict)
        
        for k, v in normalized_dict.items():
            print(f"  {k}: shape {v.shape}, min={v.min().item():.3f}, max={v.max().item():.3f}")
        # =========================================================

        # Test loading for 3 epochs
        num_epochs = 3
        print(f"Testing batch loading for {num_epochs} epochs...")
        
        device = torch.device(cfg.training.device)

        if 'train_augmentations' in cfg.task and cfg.task.train_augmentations is not None:
            augmentations = [hydra.utils.instantiate(t_cfg) for t_cfg in cfg.task.train_augmentations]
            train_augmentations = GaussianCompose(augmentations)

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                t_before_next_batch = time.time()
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                   
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    t1_1 = time.time()

                    # augmentations
                    batch = train_augmentations(batch)
                    t1_2 = time.time()
                    
                    print(f" batch generation time {t1-t_before_next_batch:.3f}")
                    print(f" device transfer time: {t1_1-t1:.3f}")
                    print(f" augmentations time: {t1_2-t1_1:.3f}")

                    t_before_next_batch = time.time()

    main()
