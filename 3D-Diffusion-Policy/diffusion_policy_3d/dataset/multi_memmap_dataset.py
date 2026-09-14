"""Memmap dataset format: the writer the converters stream into, the reader, and the
plumbing shared by every dataset that spans one or more converted memmap datasets.

The format replaces the uncompressed zarrs. Our arrays compress at ~1.1x, so zarr's
chunked reads only added chunk-granularity access, per-chunk file opens and Python
overhead on the training path (benchmarked: 14x slower batch generation than raw
memmaps served through the OS page cache, and seconds per batch on the cluster
filesystem). A dataset is a directory of raw C-order .dat files, one per key,
described by a single meta.json:

    <dataset_dir>/
      meta.json         # per-key shape/dtype, episode_ends, attrs (scene stamp, source h5s)
      point_cloud.dat
      joint_pos_action.dat
      ...

Reading opens every array as a read-only np.memmap: no RAM copy at startup, datasets
larger than RAM just work, and the OS page cache keeps the working set in RAM
automatically.

MultiMemmapDataset mirrors MultiZarrDataset 1:1 (global train/val split over all
directories, per-buffer samplers, global->local index mapping); subclasses implement
_sampler_keys(), __getitem__, get_normalizer and get_episode_init_data.
"""

import copy
import json
import os
from typing import Dict, List

import numpy as np

from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, train_episode_mask)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

META_FILENAME = 'meta.json'


class MemmapDatasetWriter:
    """Streams episodes into one raw .dat file per key; meta.json is written by finalize.

    Rows are appended with plain sequential writes, so the total frame count does not
    need to be known upfront (the spatial-memory converter discovers it episode by
    episode). dtype and per-frame shape are fixed by the first append of a key and
    asserted afterwards, since silent casting would corrupt the flat binary layout.
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        assert not os.path.exists(os.path.join(out_dir, META_FILENAME)), \
            f"{out_dir} already holds a finalized memmap dataset"
        self._files = {}
        self._dtypes: Dict[str, np.dtype] = {}
        self._frame_shapes = {}
        self._num_rows = {}

    def append(self, key, episode_array):
        episode_array = np.ascontiguousarray(episode_array)
        if key not in self._files:
            self._files[key] = open(os.path.join(self.out_dir, f'{key}.dat'), 'wb')
            self._dtypes[key] = episode_array.dtype
            self._frame_shapes[key] = episode_array.shape[1:]
            self._num_rows[key] = 0
        assert episode_array.dtype == self._dtypes[key], \
            f"{key}: dtype changed from {self._dtypes[key]} to {episode_array.dtype}"
        assert episode_array.shape[1:] == self._frame_shapes[key], \
            f"{key}: frame shape changed from {self._frame_shapes[key]} to {episode_array.shape[1:]}"
        episode_array.tofile(self._files[key])
        self._num_rows[key] += len(episode_array)

    def finalize(self, episode_ends, attrs):
        for file in self._files.values():
            file.close()
        for key, num_rows in self._num_rows.items():
            assert num_rows == episode_ends[-1], \
                f"{key} holds {num_rows} rows but episode_ends[-1] is {episode_ends[-1]}"
        meta = {
            # dtype as numpy's .str ('<f4'), which pins byte order for the raw files
            'arrays': {key: {'shape': [self._num_rows[key], *self._frame_shapes[key]],
                             'dtype': self._dtypes[key].str}
                       for key in self._files},
            'episode_ends': [int(end) for end in episode_ends],
            'attrs': attrs,
        }
        with open(os.path.join(self.out_dir, META_FILENAME), 'w') as file:
            json.dump(meta, file)


def memmap_replay_buffer(dataset_path) -> ReplayBuffer:
    """ReplayBuffer over the read-only memmaps of one dataset directory. The meta dict
    holds episode_ends plus the attrs, matching ReplayBuffer's numpy backend, where the
    zarr attrs were mixed into the meta dict the same way."""
    with open(os.path.join(dataset_path, META_FILENAME)) as file:
        meta_json = json.load(file)
    meta = dict(meta_json['attrs'])
    meta['episode_ends'] = np.array(meta_json['episode_ends'], dtype=np.int64)
    data = {
        key: np.memmap(os.path.join(dataset_path, f'{key}.dat'),
                       dtype=np.dtype(spec['dtype']), mode='r', shape=tuple(spec['shape']))
        for key, spec in meta_json['arrays'].items()
    }
    return ReplayBuffer(root={'meta': meta, 'data': data})


def dataset_attrs(dataset_path) -> dict:
    """The attrs stamped into a converted dataset."""
    with open(os.path.join(dataset_path, META_FILENAME)) as file:
        return json.load(file)['attrs']


def discover_dataset_paths(dataset_path, markers=(META_FILENAME,)) -> List[str]:
    """
    Resolve dataset_path into a list of concrete dataset paths.

    Three modes:
      1. Single dataset (str with a marker file inside)   -> [dataset_path]
      2. Directory containing multiple datasets           -> sorted list of all children
      3. Explicit list of paths                           -> list(dataset_path)

    markers defaults to memmap datasets only; callers that must accept both formats
    during the transition pass (META_FILENAME, '.zgroup').
    """
    if not isinstance(dataset_path, str):
        # Mode 3: explicit list (from Hydra ListConfig or Python list)
        paths = list(dataset_path)
        for path in paths:
            assert os.path.isdir(path), f"Dataset path does not exist: {path}"
        return paths

    dataset_path = os.path.expanduser(dataset_path)

    def is_dataset(directory):
        return any(os.path.isfile(os.path.join(directory, marker)) for marker in markers)

    if is_dataset(dataset_path):
        # Mode 1: single dataset
        return [dataset_path]

    # Mode 2: parent directory containing dataset children
    children = [os.path.join(dataset_path, name) for name in sorted(os.listdir(dataset_path))
                if os.path.isdir(os.path.join(dataset_path, name))
                and is_dataset(os.path.join(dataset_path, name))]

    assert len(children) > 0, (
        f"dataset_path '{dataset_path}' is neither a dataset nor a directory "
        f"containing datasets (no {'/'.join(markers)} found in children)"
    )
    return children


class MultiMemmapDataset(BaseDataset):
    def __init__(
        self,
        dataset_path,
        horizon=1,
        n_obs_steps=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        num_train_episodes=None,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # =====================================================================
        # Resolve dataset_path into concrete dataset paths and open the buffers
        # =====================================================================
        self.dataset_paths = discover_dataset_paths(dataset_path)
        self.replay_buffers: List[ReplayBuffer] = [
            memmap_replay_buffer(path) for path in self.dataset_paths]

        # Every dataset of one task holds the same actors, so they are read once and verified.
        self.actor_keys = [k for k in self.replay_buffers[0].keys() if k.startswith('actor_pose_')]
        for i, buf in enumerate(self.replay_buffers[1:], start=1):
            assert [k for k in buf.keys() if k.startswith('actor_pose_')] == self.actor_keys, \
                f"actor_pose_* keys mismatch between dataset 0 and {i} ({self.dataset_paths[i]})"

        # The keys the samplers materialize per sample. Subclasses build them from
        # self.actor_keys, which is why the hook runs only once the buffers are open.
        self.sampler_keys = self._sampler_keys()

        # =====================================================================
        # Compute train/val masks GLOBALLY across all buffers
        # =====================================================================
        self.episode_counts = [buf.n_episodes for buf in self.replay_buffers]
        self.total_episodes = sum(self.episode_counts)
        self._episode_cumcounts = np.cumsum(self.episode_counts)

        # EVERY episode that is not trained on is validation -- training on one trajectory
        # validates on all others, and num_train_episodes=None trains on everything with no
        # validation. This keeps the validation loss and the validation rollouts on the SAME
        # episodes: both read from this mask, the loss through per_buffer_val_masks and the
        # env runner through the global mask that get_validation_dataset swaps in.
        global_train_mask = train_episode_mask(
            n_episodes=self.total_episodes,
            num_train_episodes=num_train_episodes,
            seed=seed)
        global_val_mask = ~global_train_mask
        self.global_val_mask = global_val_mask

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

        print(f"[{type(self).__name__}] Loaded {len(self.replay_buffers)} memmap dataset(s), "
              f"{self.total_episodes} total episodes, "
              f"{sum(self._sampler_lengths)} training samples")

    def _sampler_keys(self) -> List[str]:
        """The data keys every SequenceSampler materializes for a sample."""
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

    def _episode_frame_range(self, global_episode_idx):
        """Map a global episode index to the buffer holding it and its (start, end) frame rows."""
        buf_idx, local_ep_idx = self._global_episode_to_local(global_episode_idx)
        buf = self.replay_buffers[buf_idx]
        start_idx = int(buf.episode_ends[local_ep_idx - 1]) if local_ep_idx > 0 else 0
        end_idx = int(buf.episode_ends[local_ep_idx])
        return buf, start_idx, end_idx

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
        # the env runner draws rollout init episodes from global_train_mask, so the validation
        # set advertises its own episodes there -- the ones its samplers use
        val_set.global_train_mask = self.global_val_mask
        # Swap masks so get_episode_init_data picks from val episodes
        val_set.per_buffer_train_masks = self.per_buffer_val_masks
        val_set.per_buffer_val_masks = self.per_buffer_train_masks
        return val_set

    def __len__(self) -> int:
        return int(self._cumulative_lengths[-1]) if len(self._cumulative_lengths) > 0 else 0
