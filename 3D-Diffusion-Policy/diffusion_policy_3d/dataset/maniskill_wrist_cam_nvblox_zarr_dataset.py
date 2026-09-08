"""Dataset for the mindmap baseline: nvblox feature-cloud zarr (from
scripts/dataset/conversion/convert_wrist_cam_gsworld_to_nvblox_mindmap.py) with
mindmap's keypose semantics.

Replicates mindmap's IsaacLabDataset sampling (use_keyposes=True,
only_sample_keyposes=False): every frame t is a sample; gripper_history = the
last num_history keyposes at-or-before t (front-padded with frame 0);
action/gt = the next prediction_horizon keyposes after t (back-padded with the
last keypose). Closedness comes from grasp-event intervals, not instantaneous
jaw positions (their offline estimator).

The zarr stores num_stored_vertices rows per frame (a superset, zero-padded
where the mesh was smaller; see config/scene_representation/nvblox.yaml).
__getitem__ replicates mindmap's dataloader transforms: strip the padding,
apply GeometryAugmentor (train only -- ONE random SE(3) per sample to cloud +
gripper history + gt keyposes together), then sample down to num_vertices
fresh on every access, so each epoch sees a different vertex subset.

Returns world-frame data -- the DiffuserActor normalizes internally by the
fixed workspace bounds, so get_normalizer() is identity.
"""

from typing import Dict, List

import numpy as np
import torch
import zarr

from diffusion_policy_3d.common.sampler import train_episode_mask
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy_3d.model.mindmap.embodiments.arm.estimator import (
    ArmEmbodimentOfflineEstimator,
)
from diffusion_policy_3d.model.mindmap.embodiments.arm.keypose_estimation import (
    ArmEmbodimentKeyposeEstimator,
)
from diffusion_policy_3d.model.mindmap.embodiments.arm.robot_state import (
    ArmEmbodimentRobotState,
)
from diffusion_policy_3d.model.mindmap.keyposes.keypose_detection_mode import (
    KeyposeDetectionMode,
)
from diffusion_policy_3d.model.mindmap.sample_transformer import GeometryAugmentor
from diffusion_policy_3d.model.mindmap.vertex_sampling import (
    VertexSamplingMethod,
    sample_to_n_vertices,
)


class WristCamNvbloxManiskillDataset(BaseDataset):
    def __init__(
        self,
        zarr_path: str,
        num_history: int = 3,
        prediction_horizon: int = 1,
        extra_keyposes_around_grasp_events: List[int] = (5,),
        keypose_detection_mode: str = "HIGHEST_Z_BETWEEN_GRASP",
        num_vertices: int = 2048,
        vertex_sampling_method: str = "random_without_replacement",
        apply_random_transforms: bool = False,
        random_translation_range_m: List[List[float]] = None,
        random_rpy_range_deg: List[List[float]] = None,
        seed: int = 42,
        num_train_episodes: int = None,
        val_set: bool = False,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.num_history = num_history
        self.prediction_horizon = prediction_horizon
        self.num_vertices = num_vertices
        self.vertex_sampling_method = VertexSamplingMethod(vertex_sampling_method)
        if apply_random_transforms:
            assert random_translation_range_m is not None and random_rpy_range_deg is not None
            self._augmentor = GeometryAugmentor(random_translation_range_m, random_rpy_range_deg)
        else:
            self._augmentor = None

        root = zarr.open_group(zarr_path, mode="r")
        meta = dict(root["meta"].attrs)
        assert "workspace_bounds" in meta, (
            f"{zarr_path} lacks meta.attrs['workspace_bounds'] -- "
            "reconvert with the current converter."
        )
        self.meta = meta
        num_stored_vertices = root["data/vertices"].shape[1]
        assert num_vertices <= num_stored_vertices, (
            f"num_vertices={num_vertices} exceeds the stored superset "
            f"({num_stored_vertices}) -- reconvert with a larger num_stored_vertices."
        )

        episode_ends = root["meta/episode_ends"][:]
        starts = np.concatenate([[0], episode_ends[:-1]])
        n_episodes = len(episode_ends)

        # Episode-level train/val split through the same function as the multi-dataset
        # base (multi_memmap_dataset.py), so every baseline trains on identical episode
        # sets for the same seed / num_train_episodes. As there, every episode not
        # trained on is validation.
        train_mask = train_episode_mask(n_episodes, num_train_episodes=num_train_episodes, seed=seed)
        split_mask = ~train_mask if val_set else train_mask
        self.episodes = np.flatnonzero(split_mask).tolist()
        assert self.episodes, "no episodes in split"
        # Episode-membership mask over ALL episodes (train.py logs its indices and
        # the env runner samples rollout init states from it).
        self.global_train_mask = split_mask
        # Full (unsplit) episode boundaries, for get_episode_init_data.
        self._episode_ends_all = episode_ends
        self._episode_starts_all = starts

        # Load into RAM (features stay float16 until __getitem__).
        print(f"[{'val' if val_set else 'train'}] loading {len(self.episodes)}/"
              f"{n_episodes} episodes from {zarr_path} ...")
        self._vertices, self._features, self._valid = {}, {}, {}
        self._policy_states, self._keyposes = {}, {}
        self._index = []  # (episode, local_t)
        kp_estimator = ArmEmbodimentKeyposeEstimator()
        offline_estimator = ArmEmbodimentOfflineEstimator()
        mode = KeyposeDetectionMode[keypose_detection_mode] \
            if keypose_detection_mode.isupper() else KeyposeDetectionMode(keypose_detection_mode)

        for e in self.episodes:
            s, t_end = int(starts[e]), int(episode_ends[e])
            self._vertices[e] = root["data/vertices"][s:t_end]
            self._features[e] = root["data/vertex_features"][s:t_end]
            self._valid[e] = root["data/vertices_valid_mask"][s:t_end]
            tcp = root["data/tcp_pose_proprio"][s:t_end]
            jaws = root["data/joint_pos_proprio"][s:t_end][:, -2:]

            robot_states = [
                ArmEmbodimentRobotState(
                    W_t_W_Eef=torch.from_numpy(tcp[i, :3]),
                    q_wxyz_W_Eef=torch.from_numpy(tcp[i, 3:7]),
                    gripper_jaw_positions=torch.from_numpy(jaws[i]),
                )
                for i in range(t_end - s)
            ]
            keyposes = np.asarray(
                kp_estimator.extract_keypose_indices(
                    robot_states,
                    list(extra_keyposes_around_grasp_events),
                    mode,
                ),
                dtype=np.int64,
            )
            policy_states = offline_estimator.policy_states_from_robot_states(
                robot_states, use_keyposes=True
            )
            self._policy_states[e] = torch.stack(
                [p.to_tensor() for p in policy_states]
            ).float()  # (T, 8)
            self._keyposes[e] = keyposes
            self._index.extend((e, t) for t in range(t_end - s))

        n_frames = len(self._index)
        print(f"[{'val' if val_set else 'train'}] {n_frames} samples, "
              f"~{sum(f.nbytes for f in self._features.values())/1e9:.1f} GB features in RAM")

        # augmentations off for val (mindmap's get_data_loader_without_augmentations)
        self._val_ctor_kwargs = dict(
            zarr_path=zarr_path, num_history=num_history,
            prediction_horizon=prediction_horizon,
            extra_keyposes_around_grasp_events=extra_keyposes_around_grasp_events,
            keypose_detection_mode=keypose_detection_mode,
            num_vertices=num_vertices,
            vertex_sampling_method=vertex_sampling_method,
            apply_random_transforms=False,
            seed=seed, num_train_episodes=num_train_episodes,
        )
        self._is_val = val_set

    @property
    def workspace_bounds(self) -> np.ndarray:
        """Data-derived normalization/crop bounds ([[min_xyz],[max_xyz]]), stamped
        by the converter. Source of truth for the policy's set_workspace_bounds."""
        return np.asarray(self.meta["workspace_bounds"], dtype=np.float32)

    def get_episode_init_data(self, global_episode_idx: int):
        """Init state + expert trajectory of a dataset episode, for the env runner
        to reproduce initial conditions in rollouts (gs-zarr-dataset contract:
        init_state = {'actor_poses': {'actor_pose_<name>': (7,)}, 'agent_pos': (9,)},
        expert_trajectory = (T, 9) tcp_pose + gripper jaw positions)."""
        root = zarr.open_group(self.zarr_path, mode="r")
        start = int(self._episode_starts_all[global_episode_idx])
        end = int(self._episode_ends_all[global_episode_idx])
        tcp_pose = root["data/tcp_pose_proprio"][start:end]
        qpos = root["data/joint_pos_proprio"][start:end]
        expert_trajectory = np.concatenate([tcp_pose, qpos[:, -2:]], axis=-1)

        actor_keys = [k for k in root["data"] if k.startswith("actor_pose_")]
        init_state = {
            "actor_poses": {k: root[f"data/{k}"][start] for k in actor_keys},
            "agent_pos": qpos[0],
        }
        return init_state, expert_trajectory

    def get_validation_dataset(self) -> "WristCamNvbloxManiskillDataset":
        assert not self._is_val
        return WristCamNvbloxManiskillDataset(val_set=True, **self._val_ctor_kwargs)

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # DiffuserActor normalizes internally by fixed workspace bounds.
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self):
        return len(self._index)

    def _history_and_future(self, e: int, t: int):
        """mindmap's keypose-anchored history/future (IsaacLabDataset semantics)."""
        keyposes = self._keyposes[e]
        states = self._policy_states[e]

        hist_idx = keyposes[keyposes <= t][-self.num_history:]
        n_missing = self.num_history - len(hist_idx)
        if n_missing > 0:
            hist_idx = np.concatenate([np.zeros(n_missing, dtype=np.int64), hist_idx])

        fut_idx = keyposes[keyposes > t][: self.prediction_horizon]
        n_missing = self.prediction_horizon - len(fut_idx)
        if n_missing > 0:
            fut_idx = np.concatenate(
                [fut_idx, np.full(n_missing, keyposes[-1], dtype=np.int64)]
            )
        return states[hist_idx], states[fut_idx]  # (nhist, 8), (npred, 8)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        e, t = self._index[idx]
        history, future = self._history_and_future(e, t)

        # Strip the fixed-size storage padding first (upstream's loader gets the
        # variable-length full mesh here), so the sampler below can never pick a
        # padded zero row and mark it valid.
        stored_rows = self._valid[e][t]
        vertices = torch.from_numpy(self._vertices[e][t][stored_rows])
        features = torch.from_numpy(self._features[e][t][stored_rows].astype(np.float32))

        if self._augmentor is not None:
            self._augmentor.reset()
            vertices = self._augmentor(vertices)
            history = self._augmentor(history)
            future = self._augmentor(future)

        # Fresh subset every access (train); val gets a fixed per-sample seed so
        # validation_loss stays comparable across epochs (the manual_seed inside
        # only touches the val worker's RNG).
        vertices, features, valid_mask = sample_to_n_vertices(
            vertices, features, self.num_vertices, self.vertex_sampling_method,
            seed=idx if self._is_val else None,
        )
        return {
            "obs": {
                "vertices": vertices,
                "vertex_features": features,
                "vertices_valid_mask": valid_mask,
                "gripper_history": history,
            },
            "action": future,
        }
