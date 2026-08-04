"""Plumbing shared by every dataset that spans one or more converted zarrs.

Conversion runs as a SLURM job array, so one recording session produces several sibling
zarrs. Such a dataset holds one ReplayBuffer and one SequenceSampler per zarr, computes the
train/val split GLOBALLY over all episodes of all zarrs and then slices it per buffer, and
maps the global sample / episode indices the DataLoader and the env runner hand it onto the
buffer they belong to.

Subclasses implement _sampler_keys(), __getitem__, get_normalizer and get_episode_init_data.
"""

import copy
import os
from typing import List

import numpy as np

from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


class MultiZarrDataset(BaseDataset):
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
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # =====================================================================
        # Resolve zarr_path into concrete dataset paths
        # =====================================================================
        self.zarr_paths = discover_zarr_paths(zarr_path)

        # =====================================================================
        # Open all replay buffers
        # =====================================================================
        self.replay_buffers: List[ReplayBuffer] = []
        for path in self.zarr_paths:
            # buf = ReplayBuffer.create_from_path(path, mode='r')
            buf = ReplayBuffer.copy_from_path(path)

            self.replay_buffers.append(buf)

        # Every zarr of one task holds the same actors, so they are read once and verified.
        self.actor_keys = [k for k in self.replay_buffers[0].keys() if k.startswith('actor_pose_')]
        for i, buf in enumerate(self.replay_buffers[1:], start=1):
            assert [k for k in buf.keys() if k.startswith('actor_pose_')] == self.actor_keys, \
                f"actor_pose_* keys mismatch between dataset 0 and {i} ({self.zarr_paths[i]})"

        # The keys the samplers materialize per sample. Subclasses build them from
        # self.actor_keys, which is why the hook runs only once the buffers are open.
        self.sampler_keys = self._sampler_keys()

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
                keys=self.sampler_keys
            ))

        # Precompute cumulative sample counts for O(log N) global→local index mapping
        self._sampler_lengths = [len(s) for s in self.samplers]
        self._cumulative_lengths = np.cumsum(self._sampler_lengths)

        # env runner samples rollout init episodes from this mask
        self.global_train_mask = global_train_mask

        print(f"[{type(self).__name__}] Loaded {len(self.replay_buffers)} zarr dataset(s), "
              f"{self.total_episodes} total episodes, "
              f"{sum(self._sampler_lengths)} training samples")

    def _sampler_keys(self) -> List[str]:
        """The zarr keys every SequenceSampler materializes for a sample."""
        raise NotImplementedError()

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
                keys=self.sampler_keys,
            ))
        val_set._sampler_lengths = [len(s) for s in val_set.samplers]
        val_set._cumulative_lengths = np.cumsum(val_set._sampler_lengths)
        val_set.global_train_mask = ~self.global_train_mask
        # Swap masks so get_episode_init_data picks from val episodes
        val_set.per_buffer_train_masks = self.per_buffer_val_masks
        val_set.per_buffer_val_masks = self.per_buffer_train_masks
        return val_set

    def __len__(self) -> int:
        return int(self._cumulative_lengths[-1]) if len(self._cumulative_lengths) > 0 else 0


def discover_zarr_paths(zarr_path) -> List[str]:
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
