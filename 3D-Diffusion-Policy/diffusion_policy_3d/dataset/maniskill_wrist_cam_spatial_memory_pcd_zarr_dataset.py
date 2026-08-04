"""Dataset for the spatial_memory_pcd baseline.

Loads the zarr(s) written by scripts/dataset/conversion/
convert_wrist_cam_gsworld_to_spatial_memory_pcd.py, which stores each frame's FULL
accumulated memory produced by the shared SpatialMemoryPcdSceneMapper, padded to a fixed
row count and accompanied by num_valid_points, plus the proprio and action keys shared with
the other converters. Several converted datasets are pooled through MultiZarrDataset, since
conversion runs as a job array producing one zarr per recording.

The dataset hands the padded memory over unchanged, together with num_valid_points. Reducing
it to the num_points the policy consumes is the job of the PointCloudFPS augmentation
configured in the task, which runs batched on the GPU: farthest-point sampling of a single
sample costs milliseconds on the CPU, which the dataloader workers cannot hide, whereas one
call for the whole batch costs about as much on the GPU. Training draws a fresh subset every
epoch, validation a fixed one, which the augmentation controls through random_start_point.

Plain DP3 on abs_joint_pos only -- the pcd obs-wrapper builder rejects
relative_ee_pose (RelativeEEControlWrapper only transforms gs_* keys).
"""

import copy
from typing import Dict

import numpy as np
import torch

from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.multi_zarr_dataset import MultiZarrDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class WristCamSpatialMemoryPCDManiskillDataset(MultiZarrDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            ):
        # Opens one replay buffer and one sampler per zarr (the whole dataset is loaded into
        # RAM, point_cloud is (T, N, 3) f32), splits train/val globally over all of them and
        # provides the global->local index mapping (dataset/multi_zarr_dataset.py).
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

        # Datasets converted with different mapper params must not be mixed, for the reason the
        # env-runner builder asserts the stamp against the composed config: the recorded clouds
        # would come from different scene representations.
        self.scene_representation = None
        for path, buf in zip(self.zarr_paths, self.replay_buffers):
            meta = buf.root['meta']
            attrs = meta if isinstance(meta, dict) else meta.attrs
            assert 'scene_representation' in attrs, (
                f"{path} lacks meta.attrs scene_representation -- reconvert it with the "
                "current convert_wrist_cam_gsworld_to_spatial_memory_pcd.py.")
            if self.scene_representation is None:
                self.scene_representation = attrs['scene_representation']
            assert attrs['scene_representation'] == self.scene_representation, (
                f"{path} was converted with a different scene_representation than "
                f"{self.zarr_paths[0]} -- mixing them in one training run is invalid.\n"
                f"  {self.zarr_paths[0]}: {self.scene_representation}\n  {path}: "
                f"{attrs['scene_representation']}")

    def _sampler_keys(self):
        return (['point_cloud', 'num_valid_points', 'joint_pos_proprio',
                 'joint_pos_action'] + self.actor_keys)

    def get_episode_init_data(self, global_episode_idx):
        """Init state + expert trajectory of one episode, used by the env runner to
        reproduce dataset initial conditions for rollouts."""
        buf_idx, local_ep_idx = self._global_episode_to_local(global_episode_idx)
        replay_buffer = self.replay_buffers[buf_idx]

        episode_ends = replay_buffer.episode_ends[:]
        start_idx = int(episode_ends[local_ep_idx - 1]) if local_ep_idx > 0 else 0
        end_idx = int(episode_ends[local_ep_idx])

        init_state = {
            'actor_poses': {k: replay_buffer[k][start_idx] for k in self.actor_keys},
            'agent_pos': replay_buffer['joint_pos_proprio'][start_idx],
        }
        expert_trajectory = replay_buffer['joint_pos_proprio'][start_idx:end_idx]
        return init_state, expert_trajectory

    def get_normalizer(self, **kwargs):
        """Stats over TRAIN episodes only: aspect-ratio-preserving center/radius scaling
        for the point cloud, per-DOF min/max for action/proprio with the gripper dims
        (already in [-1, 1]) forced to identity -- same conventions as the gsplat
        dataset's abs_joint_pos normalizer."""
        # Expand each buffer's per-episode train flag to a per-frame one by repeating each flag
        # over the length of its episode. Actions and proprio are small, so the train frames of
        # all buffers are concatenated and the stats computed over them at once.
        per_buffer_frame_indices = []
        actions = []
        agent_proprios = []
        for replay_buffer, train_mask in zip(self.replay_buffers, self.per_buffer_train_masks):
            episode_ends = replay_buffer.episode_ends[:]
            frame_mask = np.repeat(train_mask, np.diff(episode_ends, prepend=0))
            per_buffer_frame_indices.append(np.flatnonzero(frame_mask))
            actions.append(replay_buffer['joint_pos_action'][frame_mask])
            agent_proprios.append(replay_buffer['joint_pos_proprio'][frame_mask])

        action = torch.from_numpy(np.concatenate(actions))
        agent_proprio = torch.from_numpy(np.concatenate(agent_proprios))

        normalizer = LinearNormalizer()

        # --- Positions (preserving 3D aspect ratio) ---
        # The cloud is scaled by ONE radius on all three axes, so the scene keeps its true
        # proportions and only the largest axis reaches the [-1, 1] boundary.
        pos_min, pos_max, pos_mean, pos_std = self._point_cloud_stats(per_buffer_frame_indices)
        geometric_center = (pos_max + pos_min) / 2.0
        max_radius = torch.clamp((pos_max - pos_min).max() / 2.0, min=1e-4)
        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.ones_like(geometric_center) / max_radius,
            offset=-geometric_center / max_radius,
            input_stats_dict={
                'min': pos_min, 'max': pos_max, 'mean': pos_mean, 'std': pos_std
            }
        )

        # --- Actions and agent proprioception (per-DOF min/max, individual physical ranges) ---
        for key, values in [('action', action), ('agent_proprio', agent_proprio)]:
            k_min = values.min(dim=0)[0].clone()
            k_max = values.max(dim=0)[0].clone()

            # identity normalization for gripper which is already between [-1, 1]
            num_gripper_dims = 1 if key == 'action' else 2
            k_min[-num_gripper_dims:] = -1.0
            k_max[-num_gripper_dims:] = 1.0

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

    def _point_cloud_stats(self, per_buffer_frame_indices):
        """Component-wise min, max, mean and standard deviation over the valid points of the
        given frames, accumulated across all buffers. The frames are walked in blocks, because
        materializing every padded cloud of the dataset at once would copy several gigabytes,
        and the padding rows have to be excluded or they would drag the statistics towards the
        origin. Mean and standard deviation come from running sums accumulated in the same
        pass, in float64 because summing millions of coordinates in float32 loses precision."""
        pos_min = torch.full((3,), float('inf'))
        pos_max = torch.full((3,), float('-inf'))
        coordinate_sum = torch.zeros(3, dtype=torch.float64)
        squared_coordinate_sum = torch.zeros(3, dtype=torch.float64)
        num_points = 0
        for replay_buffer, frame_indices in zip(self.replay_buffers, per_buffer_frame_indices):
            num_valid_points = replay_buffer['num_valid_points']
            for start in range(0, len(frame_indices), 512):
                block = frame_indices[start:start + 512]
                clouds = replay_buffer['point_cloud'][block]
                is_valid = (np.arange(clouds.shape[1])[None, :]
                            < num_valid_points[block][:, None])
                points = torch.from_numpy(clouds[is_valid])
                pos_min = torch.minimum(pos_min, points.min(dim=0)[0])
                pos_max = torch.maximum(pos_max, points.max(dim=0)[0])
                coordinate_sum += points.to(torch.float64).sum(dim=0)
                squared_coordinate_sum += points.to(torch.float64).square().sum(dim=0)
                num_points += points.shape[0]

        pos_mean = coordinate_sum / num_points
        pos_variance = squared_coordinate_sum / num_points - pos_mean.square()
        return (pos_min, pos_max, pos_mean.to(torch.float32),
                pos_variance.clamp(min=0.0).sqrt().to(torch.float32))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        buf_idx, local_idx = self._global_sample_to_local(idx)
        sample = self.samplers[buf_idx].sample_sequence(local_idx)

        data = {
            'obs': {
                # policy only consumes the first n_obs_steps of the horizon window.
                # The memory is passed on with its padding, since PointCloudFPS reduces it to
                # num_points on the GPU and needs num_valid_points as the per-cloud length.
                'point_cloud': torch.from_numpy(sample['point_cloud'][:self.n_obs_steps]),
                'num_valid_points': torch.from_numpy(
                    sample['num_valid_points'][:self.n_obs_steps].astype(np.int64)),
                'agent_proprio': torch.from_numpy(sample['joint_pos_proprio'][:self.n_obs_steps].astype(np.float32)),
            },
            'action': torch.from_numpy(sample['joint_pos_action'].astype(np.float32)),
            # only used for reproducing rollout init states, not for learning
            'actor_poses': {k: torch.from_numpy(sample[k]) for k in self.actor_keys},
        }
        return data


if __name__ == "__main__":
    import time

    import hydra
    import tqdm
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from diffusion_policy_3d.common.pytorch_util import dict_apply
    from diffusion_policy_3d.dataset.data_augmentations import GaussianCompose

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    @hydra.main(
        version_base=None,
        config_path="../config",
        config_name="wrist_cam_spatial_memory_dp3"
    )
    def main(cfg):
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseDataset), f"dataset must be BaseDataset, got {type(dataset)}"
        print(f"Dataset instantiated. Length: {len(dataset)}")

        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        print("DataLoader instantiated.")

        device = torch.device(cfg.training.device)
        # train.py requires both lists, so instantiate them the same way instead of tolerating
        # their absence -- a config that fails here would fail at the start of training too.
        train_data_augmentations = GaussianCompose(
            [hydra.utils.instantiate(t_cfg) for t_cfg in cfg.task.train_data_augmentations])
        val_data_augmentations = GaussianCompose(
            [hydra.utils.instantiate(t_cfg) for t_cfg in cfg.task.val_data_augmentations])

        # =========================================================
        # Test normalization, on an augmented batch, since that is what the policy receives
        # =========================================================
        print("\n--- Testing Normalizer ---")
        normalizer = dataset.get_normalizer()

        batch = next(iter(train_dataloader))
        num_valid_points = batch['obs']['num_valid_points']
        print(f"Got test batch, point_cloud {tuple(batch['obs']['point_cloud'].shape)} holding "
              f"{int(num_valid_points.min())}...{int(num_valid_points.max())} accumulated points. "
              "Augmenting and normalizing...")

        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        batch = train_data_augmentations(batch)

        test_dict = {
            'point_cloud': batch['obs']['point_cloud'],
            'agent_proprio': batch['obs']['agent_proprio'],
            'action': batch['action'],
        }
        normalized_dict = normalizer.normalize(test_dict)

        for k, v in normalized_dict.items():
            print(f"  {k}: shape {tuple(v.shape)}, min={v.min().item():.3f}, max={v.max().item():.3f}")

        # Validation must present the same subset every epoch, training a different one. Both
        # augmentations consume the SAME raw batch, and a copy each time because they replace
        # point_cloud and pop num_valid_points in place.
        raw_batch = dict_apply(next(iter(train_dataloader)), lambda x: x.to(device))

        def sampled_cloud(augmentations):
            return augmentations(copy.deepcopy(raw_batch))['obs']['point_cloud']
        print(f"  train subsets vary across epochs: "
              f"{not torch.equal(sampled_cloud(train_data_augmentations), sampled_cloud(train_data_augmentations))}")
        print(f"  val subsets stay fixed:           "
              f"{torch.equal(sampled_cloud(val_data_augmentations), sampled_cloud(val_data_augmentations))}")
        # =========================================================

        num_epochs = 3
        print(f"Testing batch loading for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {epoch}",
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                t_before_next_batch = time.time()
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()

                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    t1_1 = time.time()

                    batch = train_data_augmentations(batch)
                    t1_2 = time.time()

                    print(f" batch generation time {t1-t_before_next_batch:.3f}")
                    print(f" device transfer time: {t1_1-t1:.3f}")
                    print(f" augmentations time: {t1_2-t1_1:.3f}")

                    t_before_next_batch = time.time()

    main()
