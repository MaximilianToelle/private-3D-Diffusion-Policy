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
from diffusion_policy_3d.common.sampler import SequenceSampler
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, PerTimestepLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.multi_zarr_dataset import MultiZarrDataset
from diffusion_policy_3d.common.transform_utils import pose7_to_mat_np, mat_to_pose9d_np

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


class WristCamGSManiskillDataset(MultiZarrDataset):
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
        representation_space="abs_joint_pos",  # abs_joint_pos | relative_ee_pose
        min_opacity=0.,       # Gaussians below this opacity are pruned during subsampling / normalization
        num_gaussians=1024,   # K: number of Gaussians kept per sample after CPU subsampling
        verbose=False,
    ):
        self.num_gaussians = num_gaussians
        self.verbose = verbose
        assert representation_space in ("abs_joint_pos", "relative_ee_pose"), \
            f"representation_space must be 'abs_joint_pos' or 'relative_ee_pose', got '{representation_space}'"
        self.representation_space = representation_space

        # Opens one replay buffer and one sampler per zarr, splits train/val globally over all
        # of them and provides the global->local index mapping (dataset/multi_zarr_dataset.py).
        super().__init__(
            zarr_path=zarr_path,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
        )

        # Cached array handles per buffer, read by _get_synced_obs_slice.
        self.gsplats_arrays = []
        self.joint_pos_proprio_arrays = []
        self.tcp_pose_proprio_arrays = []
        for buf in self.replay_buffers:
            self.gsplats_arrays.append(buf.root['data']['gsplats'])
            self.joint_pos_proprio_arrays.append(buf.root['data']['joint_pos_proprio'])
            self.tcp_pose_proprio_arrays.append(buf.root['data']['tcp_pose_proprio'])

        # =====================================================================
        # Read gs_params / gs_param_sizes from the first buffer, verify all match
        # =====================================================================
        first_buf = self.replay_buffers[0]
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

        self.min_opacity = min_opacity
        # Validate: config min_opacity must be >= the dataset's min_opacity.
        for i, buf in enumerate(self.replay_buffers):
            m = buf.root['meta']
            a = m if isinstance(m, dict) else m.attrs
            dataset_min_opacity = a.get('min_opacity', None)
            if dataset_min_opacity is not None and min_opacity < dataset_min_opacity:
                raise ValueError(
                    f"Config min_opacity ({min_opacity}) is below the dataset's "
                    f"min_opacity ({dataset_min_opacity}) in dataset {i} "
                    f"({self.zarr_paths[i]}). During evaluation the policy would "
                    f"see Gaussians that were pruned during training. "
                    f"Set min_opacity >= {dataset_min_opacity}."
                )

    def _sampler_keys(self):
        if self.representation_space == "abs_joint_pos":
            return ['joint_pos_action'] + self.actor_keys
        elif self.representation_space == "relative_ee_pose":
            return ['tcp_pose_action'] + self.actor_keys

    def get_episode_init_data(self, global_episode_idx):
        """
        Return init state data for a specific global episode index.
        Used by the env runner to reproduce initial conditions from the dataset.
        
        Returns:
            init_state: dict with 'actor_poses' and 'agent_pos'
            expert_trajectory: np.ndarray of the full expert trajectory.
                - abs_joint_pos: joint state (qpos) trajectory
                - relative_ee_pose: tcp_pose (7D) + gripper_state (2D) = 9D trajectory
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
        init_state['agent_pos'] = buf['joint_pos_proprio'][start_idx]
        
        if self.representation_space == "abs_joint_pos":
            expert_trajectory = buf['joint_pos_proprio'][start_idx:end_idx]
        elif self.representation_space == "relative_ee_pose":
            # tcp_pose_proprio: (T, 7) [x,y,z,qw,qx,qy,qz], gripper: last 2 dims of joint_pos_proprio
            tcp_pose_proprio = buf['tcp_pose_proprio'][start_idx:end_idx]
            gripper_state = buf['joint_pos_proprio'][start_idx:end_idx][:, -2:]
            expert_trajectory = np.concatenate([tcp_pose_proprio, gripper_state], axis=-1)  # (T, 9)
        
        return init_state, expert_trajectory

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
            'agent_proprio': {'min': None, 'max': None},
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
            norm_keys = ['joint_pos_action', 'joint_pos_proprio', 'gsplats']      # actor poses are only used for reproducing init states
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
                torch_data = self._sample_to_data(sample, gaussian_selection="opacity_filter")
                obs = torch_data['obs']

                update_min_max('action', torch_data['action'])
                update_min_max('agent_proprio', obs['agent_proprio'])
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
            M = stats['gs_log_scales']['count']
            mean_log_scales = stats['gs_log_scales']['sum'] / M
            var_log_scales = stats['gs_log_scales']['sum_sq'] / M - (mean_log_scales ** 2)
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

        # --- Actions and Agent Proprioception (separate normalization of each DOF as they have individual physical ranges!) ---
        for key in ['action', 'agent_proprio']:
            k_min = stats[key]['min']
            k_max = stats[key]['max']

            # identity normalization for gripper which is already between [-1, 1]
            if key == "action":
                k_min[-1:] = -1.0
                k_max[-1:] = 1.0
            elif key == "agent_proprio":
                k_min[-2:] = -1.0
                k_max[-2:] = 1.0
            else:
                raise ValueError(f"key {key} is not an implemented option")

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
        probe_sample['joint_pos_proprio'] = self._get_synced_obs_slice(
            self.samplers[0].indices[0], self.joint_pos_proprio_arrays[0], name="joint_pos_proprio")
        probe_sample['tcp_pose_proprio'] = self._get_synced_obs_slice(
            self.samplers[0].indices[0], self.tcp_pose_proprio_arrays[0], name="tcp_pose_proprio")
        probe_sample['gsplats'] = self._get_synced_obs_slice(
            self.samplers[0].indices[0], self.gsplats_arrays[0], name="gsplats")
        probe_data = self._sample_to_data(probe_sample, gaussian_selection="opacity_filter")
        D_agent_proprio = probe_data['obs']['agent_proprio'].shape[-1]
        D_action = probe_data['action'].shape[-1]
        D_gs_positions = 3

        # =====================================================================
        # Phase 1: Collect and sort data per dataset
        # =====================================================================
        # For each (timestep, feature_dim), we store a list of sorted 1D tensors,
        # one per dataset. This avoids holding all raw samples in memory at once.
        # Layout: sorted_chunks[t][d] = [sorted_1d_tensor_from_ds0, sorted_1d_tensor_from_ds1, ...]
        # NOTE: we do not need to normalize rotations and gripper state but only positions
        agent_proprio_sorted_chunks = [[[] for _ in range(3)] for _ in range(T_obs)]
        action_sorted_chunks = [[[] for _ in range(3)] for _ in range(T_horizon)]
        gs_positions_sorted_chunks = [[[] for _ in range(D_gs_positions)] for _ in range(T_obs)]

        for buf_i, buf in enumerate(self.replay_buffers):
            norm_mask = self.per_buffer_train_masks[buf_i]
            normalization_sampler = SequenceSampler(
                replay_buffer=buf,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=norm_mask,
                keys=self.sampler_keys,  # sampler only handles these keys
            )

            desc = f"Collecting stats from dataset {buf_i+1}/{len(self.replay_buffers)}"
            
            chunk_size = 1000
            sampler_len = len(normalization_sampler)
            
            for chunk_start in range(0, sampler_len, chunk_size):
                chunk_end = min(chunk_start + chunk_size, sampler_len)
                
                dataset_agent_proprio_vals = []
                dataset_action_vals = []
                dataset_gs_positions_vals = []
                
                for idx in tqdm(range(chunk_start, chunk_end), desc=f"{desc} (chunk {chunk_start//chunk_size + 1}/{(sampler_len + chunk_size - 1)//chunk_size})"):
                    sample = normalization_sampler.sample_sequence(idx)
                    raw_indices = normalization_sampler.indices[idx]
                    sample['joint_pos_proprio'] = self._get_synced_obs_slice(
                        raw_indices, self.joint_pos_proprio_arrays[buf_i], name="joint_pos_proprio")
                    sample['tcp_pose_proprio'] = self._get_synced_obs_slice(
                        raw_indices, self.tcp_pose_proprio_arrays[buf_i], name="tcp_pose_proprio")
                    sample['gsplats'] = self._get_synced_obs_slice(
                        raw_indices, self.gsplats_arrays[buf_i], name="gsplats")

                    # Assumption: Gaussians are prescanned -> fixed opacities,
                    # so the filtered count is identical across samples (safe to stack below).
                    torch_data = self._sample_to_data(sample, gaussian_selection="opacity_filter")

                    dataset_agent_proprio_vals.append(torch_data['obs']['agent_proprio'][..., :3])        # (T_obs, 3)
                    dataset_action_vals.append(torch_data['action'][..., :3])                     # (T_horizon, 3)
                    dataset_gs_positions_vals.append(torch_data['obs']['gs_positions'])  # (T_obs, N_gaussians, 3)

                if len(dataset_agent_proprio_vals) == 0:
                    continue

                # Stack this chunk's samples, extract per (t, d), sort, and store
                # the compact sorted 1D tensors. Then delete the raw 
                # tensors to free memory before processing the next chunk.
                agent_proprio_stacked = torch.stack(dataset_agent_proprio_vals, dim=0)  # (N_chunk, T_obs, D_agent_proprio)
                for t in range(T_obs):
                    for d in range(3):
                        agent_proprio_sorted_chunks[t][d].append(
                            agent_proprio_stacked[:, t, d].contiguous().sort()[0])
                del dataset_agent_proprio_vals, agent_proprio_stacked

                action_stacked = torch.stack(dataset_action_vals, dim=0)  # (N_chunk, T_horizon, D_action)
                for t in range(T_horizon):
                    for d in range(3):
                        action_sorted_chunks[t][d].append(
                            action_stacked[:, t, d].contiguous().sort()[0])
                del dataset_action_vals, action_stacked

                # gs_positions: each sample has (T_obs, N_gaussians, 3) with fixed N_gaussians.
                # We stack along a new sample dimension, then flatten over both samples and
                # Gaussians to get all values for a given (t, d).
                gs_positions_stacked = torch.stack(dataset_gs_positions_vals, dim=0)  # (N_chunk, T_obs, N_gaussians, 3)
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

        # --- agent_proprio ---
        p02_agent_proprio = torch.full((T_obs, D_agent_proprio), -1.0)
        p98_agent_proprio = torch.full((T_obs, D_agent_proprio), 1.0)

        p02_agent_proprio_pos, p98_agent_proprio_pos = build_percentile_tensors(
            agent_proprio_sorted_chunks, T_obs, 3, lower_percentile, upper_percentile)
        
        p02_agent_proprio[:, :3] = p02_agent_proprio_pos
        p98_agent_proprio[:, :3] = p98_agent_proprio_pos

        normalizer['agent_proprio'] = PerTimestepLinearNormalizer.create_clamped_percentile_normalizer(
            p02=p02_agent_proprio, p98=p98_agent_proprio,
            n_timesteps=T_obs, n_features=D_agent_proprio,
        )
        del agent_proprio_sorted_chunks

        # --- action ---
        p02_action = torch.full((self.horizon, D_action), -1.0)
        p98_action = torch.full((self.horizon, D_action), 1.0)

        p02_action_pos, p98_action_pos = build_percentile_tensors(
            action_sorted_chunks, T_horizon, 3, lower_percentile, upper_percentile)
        
        p02_action[:, :3] = p02_action_pos
        p98_action[:, :3] = p98_action_pos

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

    def _get_synced_obs_slice(self, raw_indices, array, name=""):
        """
        Read the first `n_obs_steps` observation frames for one training sample.

        Key Idea:
          The SequenceSampler hands us a window sized for the ACTION horizon (`horizon`
          frames, e.g. 16), because the policy predicts a full horizon of actions. But the
          policy only CONDITIONS on a short observation history (`n_obs_steps` frames, e.g. 2).
          So instead of loading all 16 heavy Gaussian frames and throwing 14 away, we read
          only the first `n_obs_steps` of that window. 

        THE four indices (from SequenceSampler.create_indices) describe the horizon window:
          - array_start / array_end : the physical row range in `array` that actually
            exists on this episode (already clamped to episode bounds by the sampler).
          - pad_before / pad_after  : the window is `[pad_before : pad_after]` filled with real
            data; slots before `pad_before` and from `pad_after` on are out-of-episode and must
            be edge-padded (repeat first/last real frame). `pad_before > 0` means the window
            starts before the episode (near t=0, no history); `pad_after < horizon` means it
            runs past the episode end.

          We reinterpret these for the SHORTER observation window of length `n_obs_steps`:
          `pad_before` is exactly the number of leading padding slots our obs window also needs
          (both windows share the same slot-0 logical time), and real observation frames begin
          at physical row `array_start`.
        """
        t_start = time.time()

        array_start, array_end, pad_before, pad_after = raw_indices

        # How many REAL frames must we actually read from `array`?
        # We want `n_obs_steps` output frames; the first `pad_before` of them are start-padding
        # (no history), so only `n_obs_steps - pad_before` need real data. Floor at 1 so we
        # always read at least one frame to pad from, and never read past the episode's real
        # data (`array_end - array_start`).
        num_frames_to_read = max(1, self.n_obs_steps - pad_before)
        num_frames_to_read = min(num_frames_to_read, array_end - array_start)

        t_before_read = time.time()
        real_frames = array[array_start : array_start + num_frames_to_read]
        t_after_read = time.time()

        # Output buffer, one row per observation step (missing rows get padded below).
        obs_slice = np.zeros(
            (self.n_obs_steps,) + real_frames.shape[1:],
            dtype=real_frames.dtype,
        )

        # --- Start-of-episode padding: fill the leading `pad_before` slots (no history exists)
        # with the first real frame (edge replication backwards in time). ---
        if pad_before > 0:
            pad_count = min(pad_before, self.n_obs_steps)
            obs_slice[:pad_count] = real_frames[0]

        # Real frames: drop them into their chronological slots, right after the start padding. 
        if pad_before < self.n_obs_steps:
            num_real = min(self.n_obs_steps - pad_before, num_frames_to_read)
            insert_end = pad_before + num_real
            obs_slice[pad_before : insert_end] = real_frames[:num_real]

            # End-of-episode padding: if the episode ended before filling all
            # `n_obs_steps` slots, repeat the last real frame forwards in time.
            if insert_end < self.n_obs_steps:
                obs_slice[insert_end:] = real_frames[-1]

        t_end = time.time()
        if self.verbose:
            pid = os.getpid()
            print(
                f"    [Worker {pid}] _get_synced_obs_slice ({name}):\n"
                f"      - Index extraction: {(t_before_read - t_start) * 1000:.3f} ms\n"
                f"      - Array Read:       {(t_after_read - t_before_read) * 1000:.3f} ms\n"
                f"      - Copy & Padding:   {(t_end - t_after_read) * 1000:.3f} ms\n"
                f"      - Total:            {(t_end - t_start) * 1000:.3f} ms"
            )
        return obs_slice

    def _opacity_filter_indices(self, gsplats_torch, min_opacity):
        """
        NORMALIZATION selection: keep all Gaussians with opacity >= min_opacity across the whole
        observation window (min over the T_obs axis), active or not, with NO downsampling.

        The policy never sees Gaussians pruned below min_opacity, so computing normalization stats
        over exactly this set yields more accurate stats.

        Args:
            gsplats_torch: (T_obs, N, D) float tensor (raw channels, not yet transformed).
            min_opacity:   opacity threshold.
        Returns:
            idx: (M,) long tensor of Gaussians with opacity >= min_opacity (variable M),
                 or None if this dataset has no opacity channel to filter on.
        """
        if 'opacities' not in self.param_slices:
            return None
        opacities = gsplats_torch[..., self.param_slices['opacities']].amin(dim=0).squeeze(-1)  # (N,)
        keep_mask = opacities >= min_opacity
        return torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)

    def _subsample_gaussian_indices(self, gsplats_torch, min_opacity, num_gaussians):
        """
        BATCH-LOADING selection: pick `num_gaussians` (K) valid Gaussians per sample, on
        the CPU, so only K Gaussians are transformed and crossed to the GPU.

        A Gaussian is valid if it is active and has opacity >= min_opacity across the WHOLE
        observation window (min over the T_obs axis). Among the valid ones we draw K uniformly
        without replacement via `topk` over random keys.

        Args:
            gsplats_torch: (T_obs, N, D) float tensor.
            min_opacity:   opacity threshold.
            num_gaussians: K, the number of Gaussians to keep.
        Returns:
            idx: (K,) long tensor of selected Gaussian indices, or None if this dataset has
                 neither an opacity nor an active-mask channel to filter on.
        """
        valid_mask = None
        if 'opacities' in self.param_slices:
            opacities = gsplats_torch[..., self.param_slices['opacities']].amin(dim=0).squeeze(-1)
            valid_mask = (opacities >= min_opacity)
        if 'active_gaussians_mask' in self.param_slices:
            active = gsplats_torch[..., self.param_slices['active_gaussians_mask']].amin(dim=0).squeeze(-1)
            active_mask = (active > 0.5)
            valid_mask = active_mask if valid_mask is None else (valid_mask & active_mask)

        if valid_mask is None:
            return None

        # Gather valid Gaussians, draw K random keys over ONLY those (cheaper than over all N and
        # avoids a full-length sentinel array), then map the local picks back to global indices.
        valid_indices = torch.where(valid_mask)[0]
        num_valid = valid_indices.shape[0]
        assert num_valid >= num_gaussians, (
            f"Not enough valid Gaussians to sample from: "
            f"{num_valid} valid < {num_gaussians} requested (min_opacity={min_opacity})."
        )
        rand_keys = torch.rand(num_valid)
        _, topk_local = torch.topk(rand_keys, num_gaussians, largest=False)
        return valid_indices[topk_local]

    def _sample_to_data(self, sample, gaussian_selection=None):
        """
        Returns data as dict of torch tensors.
        Handles two representation modes:
          - abs_joint_pos: pass-through (joint positions as proprio and actions)
          - relative_ee_pose: SE(3) transform GS into anchor frame, relative proprio and actions

        `gaussian_selection` chooses which Gaussians survive (applied BEFORE the SE(3) transform,
        so only the survivors are transformed and later transferred to the GPU). All variants use
        the dataset's own `self.min_opacity` / `self.num_gaussians`:
          - "subsample"      (used by __getitem__): active AND opacity >= min_opacity, then
                             randomly downsample to exactly self.num_gaussians. Also drops the
                             filter-only channels (opacities, active_gaussians_mask) afterwards.
          - "opacity_filter" (used by get_normalizer): keep ALL Gaussians with
                             opacity >= min_opacity (active or not), no downsample.
          - None             (default): keep every Gaussian (legacy behavior).
        """
        assert gaussian_selection in (None, "subsample", "opacity_filter"), \
            f"invalid gaussian_selection: {gaussian_selection}"

        gsplats_torch = torch.from_numpy(sample['gsplats']).float()   # (n_obs_steps, N, D)

        # subsampling is done at the beginning for consistency between both representations
        # it especially benefits relative_ee_pose as subsequent transformations are only applied to self.num_gaussians (K)
        if gaussian_selection == "subsample":
            keep_idx = self._subsample_gaussian_indices(
                gsplats_torch, self.min_opacity, self.num_gaussians)
        elif gaussian_selection == "opacity_filter":
            keep_idx = self._opacity_filter_indices(gsplats_torch, self.min_opacity)
        else:
            keep_idx = None
        if keep_idx is not None:
            gsplats_torch = gsplats_torch[:, keep_idx, :]              # (n_obs_steps, K, D)

        # When subsampling, opacities/active_gaussians_mask were consumed by the filter and are
        # not read by the model, so drop them from the produced obs.
        drop_gs_params = {'opacities', 'active_gaussians_mask'} if gaussian_selection == "subsample" else set()

        if self.representation_space == "abs_joint_pos":
            # =====================================================
            # Absolute joint position mode
            # =====================================================
            agent_proprio = torch.from_numpy(sample['joint_pos_proprio'])
            action = torch.from_numpy(sample['joint_pos_action'])

            obs_dict = {'agent_proprio': agent_proprio}

            # Dynamically populate obs_dict based on gs_params layout
            for p, size in zip(self.gs_params, self.gs_param_sizes):
                if p in drop_gs_params:
                    continue
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
            
            tcp_pose_proprio = sample['tcp_pose_proprio'].astype(np.float32)   # (n_obs_steps, 7)
            tcp_pose_action = sample['tcp_pose_action'].astype(np.float32)     # (horizon, 8) = 7D pose + 1D gripper
            joint_pos_proprio = sample['joint_pos_proprio'].astype(np.float32)  # (n_obs_steps, 9)

            # --- Anchor frame: TCP at the last observation step ---
            T_anchor_tcp_to_base = pose7_to_mat_np(tcp_pose_proprio[-1])      # (4, 4)
            T_anchor_base_to_tcp = np.linalg.inv(T_anchor_tcp_to_base)           # (4, 4)
            R_anchor_base_to_tcp = T_anchor_base_to_tcp[:3, :3]                # (3, 3)
            t_anchor_base_to_tcp = T_anchor_base_to_tcp[:3, 3]                 # (3,)

            # --- Transform the (already selected) GS observations into the anchor frame ---
            R_anchor_base_to_tcp_torch = torch.from_numpy(R_anchor_base_to_tcp)   # (3, 3)
            t_anchor_base_to_tcp_torch = torch.from_numpy(t_anchor_base_to_tcp)   # (3,)

            obs_dict = {}
            for p, size in zip(self.gs_params, self.gs_param_sizes):
                if p in drop_gs_params:
                    continue
                obs_key = self.param_to_obs_key.get(p, f"gs_{p}")
                s = self.param_slices[p]
                group = gsplats_torch[..., s]                                 # strided view into gsplats_torch

                if p == 'positions':
                    # (T_obs, K, 3) @ (3, 3) + (3,)
                    obs_dict[obs_key] = group @ R_anchor_base_to_tcp_torch.T + t_anchor_base_to_tcp_torch
                elif p == 'rotations_9d':
                    T_obs, K = group.shape[:2]
                    rot_mats = group.reshape(T_obs, K, 3, 3)                  # (T_obs, K, 3, 3)
                    rot_mats = R_anchor_base_to_tcp_torch @ rot_mats
                    obs_dict[obs_key] = rot_mats.reshape(T_obs, K, 9)
                elif p == 'surf_normals':
                    obs_dict[obs_key] = group @ R_anchor_base_to_tcp_torch.T
                else:
                    # Untouched channels (semantics, log_scales, rgbs, ...): materialize a
                    # contiguous tensor so downstream gather / pin_memory / device-transfer is clean
                    obs_dict[obs_key] = group.contiguous()

            if 'positions' in self.gs_params:
                # point_cloud is consumed by the baseline (non-gsplat) DP3 policy
                obs_dict['point_cloud'] = obs_dict[self.param_to_obs_key['positions']]

            # --- Proprioception: relative EE pose + gripper ---
            # pos(3) + rot6d(6) + gripper(2) = 11D per obs step
            T_tcp_pose_proprio_to_base = pose7_to_mat_np(tcp_pose_proprio)   # (n_obs_steps, 4, 4)
            T_relative_tcp = T_anchor_base_to_tcp[None, :, :] @ T_tcp_pose_proprio_to_base   # (n_obs_steps, 4, 4)
            rel_tcp_pose9d = mat_to_pose9d_np(T_relative_tcp)  # (n_obs_steps, 9) 
            gripper_pos = joint_pos_proprio[:, -2:]          # (n_obs_steps, 2), absolute gripper position
            agent_proprio_np = np.concatenate([rel_tcp_pose9d, gripper_pos], axis=-1)  # (n_obs_steps, 11)
            obs_dict['agent_proprio'] = torch.from_numpy(agent_proprio_np)

            # --- 3. Actions: relative EE pose + gripper ---
            # pos(3) + rot6d(6) + gripper(1) = 10D per action step
            
            # Split tcp_pose_action into pose (7D) and gripper (1D)
            tcp_pose_action_7d = tcp_pose_action[:, :7]   # (horizon, 7)
            gripper_action = tcp_pose_action[:, 7:]     # (horizon, 1)
            
            T_tcp_pose_action_to_base = pose7_to_mat_np(tcp_pose_action_7d)   # (horizon, 4, 4)
            T_rel_action = T_anchor_base_to_tcp[None, :, :] @ T_tcp_pose_action_to_base # (horizon, 4, 4)
            rel_action_pose9d = mat_to_pose9d_np(T_rel_action)  # (horizon, 9)
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
        
        # needed in relative_ee_pose for gripper state!
        sample['joint_pos_proprio'] = self._get_synced_obs_slice(raw_indices, self.joint_pos_proprio_arrays[buf_idx], name="joint_pos_proprio")
        if self.representation_space == "relative_ee_pose":
            sample['tcp_pose_proprio'] = self._get_synced_obs_slice(raw_indices, self.tcp_pose_proprio_arrays[buf_idx], name="tcp_pose_proprio")
        sample['gsplats'] = self._get_synced_obs_slice(raw_indices, self.gsplats_arrays[buf_idx], name="gsplats")
        t_6 = time.time()

        data = self._sample_to_data(sample, gaussian_selection="subsample")
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
            'agent_proprio': batch['obs']['agent_proprio'],
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

        if 'train_data_augmentations' in cfg.task and cfg.task.train_data_augmentations is not None:
            augmentations = [hydra.utils.instantiate(t_cfg) for t_cfg in cfg.task.train_data_augmentations]
            train_data_augmentations = GaussianCompose(augmentations)

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
                    batch = train_data_augmentations(batch)
                    t1_2 = time.time()
                    
                    print(f" batch generation time {t1-t_before_next_batch:.3f}")
                    print(f" device transfer time: {t1_1-t1:.3f}")
                    print(f" augmentations time: {t1_2-t1_1:.3f}")

                    t_before_next_batch = time.time()

    main()
