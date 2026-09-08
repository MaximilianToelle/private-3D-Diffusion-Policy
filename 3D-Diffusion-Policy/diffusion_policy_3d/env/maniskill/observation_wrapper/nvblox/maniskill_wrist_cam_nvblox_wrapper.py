"""Wrist-cam nvblox spatial-memory perception wrapper (mindmap baseline).

Stateful: every env step integrates the wrist camera's depth/rgb into the
persistent nvblox TSDF+feature map (robot pixels masked out), and the emitted
observation is the featurized vertex cloud sampled from the CURRENT map -- the
map itself carries the history, so the policy needs no frame stacking.

Robot masking uses the live SAPIEN ids read by maniskill_scene_infos.py. No GSWorld
semantic constants are involved here (those only apply to the recorded h5 files,
which the GS wrapper remapped).

Observation keys (matching the training dataset contract):
  vertices             (num_vertices, 3)    float32, world frame
  vertex_features      (num_vertices, C)    float32
  vertices_valid_mask  (num_vertices,)      bool
"""

import torch
from gym import spaces

from diffusion_policy_3d.baseline_scene_integration.perception_utils import (
    cam2world_cv_from_extrinsic_cv,
)
from diffusion_policy_3d.baseline_scene_integration.nvblox_scene_mapper import (
    FeatureCloudSceneMapper,
)
from diffusion_policy_3d.env.maniskill.observation_wrapper.maniskill_dp3_base_obs_wrapper import (
    ManiSkillDP3BaseObsWrapper,
)
from diffusion_policy_3d.env.maniskill.observation_wrapper.maniskill_scene_infos import (
    robot_link_seg_ids,
)
from diffusion_policy_3d.model.mindmap.vertex_sampling import (
    VertexSamplingMethod,
    sample_to_n_vertices,
)


class MockNvbloxSceneMapper(FeatureCloudSceneMapper):
    """Host-side stand-in for NvbloxSceneMapper (which needs nvblox_torch, only
    available in the mindmap_baseline container). Lets the full env-runner
    plumbing -- env creation, wrapper statefulness, observation keys/shapes,
    policy contract -- be exercised without nvblox. Stands in for the mesh with
    num_mesh_vertices random vertices inside the workspace bounds."""

    def __init__(self, workspace_bounds, num_mesh_vertices=4096, embedding_dim=768,
                 device="cuda"):
        self.workspace_bounds = torch.as_tensor(workspace_bounds, dtype=torch.float32)
        self.num_mesh_vertices = num_mesh_vertices
        self._embedding_dim = embedding_dim
        self.device = device
        self.num_integrated_frames = 0
        self.num_resets = 0

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def reset(self):
        self.num_resets += 1
        self.num_integrated_frames = 0

    def integrate_frame(self, depth_m, rgb, robot_mask, intrinsics, cam2world_cv, decay=True):
        assert depth_m.shape[0] == depth_m.shape[1], "square images required"
        assert robot_mask.dtype == torch.bool
        self.num_integrated_frames += 1

    def get_scene_representation(self):
        lower, upper = self.workspace_bounds[0], self.workspace_bounds[1]
        vertices = (
            torch.rand(self.num_mesh_vertices, 3, device=self.device)
            * (upper - lower).to(self.device) + lower.to(self.device)
        )
        features = torch.randn(self.num_mesh_vertices, self.embedding_dim, device=self.device)
        return vertices, features


class WristCamNvbloxManiskillDP3Wrapper(ManiSkillDP3BaseObsWrapper):
    def __init__(
        self,
        env,
        representation_space,
        agent_proprio_dim,
        cam_name,
        scene_mapper,
        num_vertices,
        vertex_sampling_method,
    ):
        # Perception params must be set BEFORE super().__init__() (which calls
        # _perception_observation_space).
        self.cam = cam_name
        self.scene_mapper = scene_mapper
        # How many vertices the policy consumes, and how they are drawn out of the mesh.
        # Not mapper parameters: the mapper returns the whole mesh, this wrapper samples it.
        self.num_vertices = num_vertices
        self.vertex_sampling_method = VertexSamplingMethod(vertex_sampling_method)
        self.feature_dim = scene_mapper.embedding_dim

        super().__init__(env, representation_space, agent_proprio_dim, render_cam_name=cam_name)

        self._robot_seg_ids = robot_link_seg_ids(self.env)
        self._is_first_integration = True

    def _perception_observation_space(self):
        return {
            "vertices": spaces.Box(
                low=-float("inf"), high=float("inf"),
                shape=(self.num_vertices, 3), dtype="float32",
            ),
            "vertex_features": spaces.Box(
                low=-float("inf"), high=float("inf"),
                shape=(self.num_vertices, self.feature_dim), dtype="float32",
            ),
            "vertices_valid_mask": spaces.Box(
                low=0, high=1, shape=(self.num_vertices,), dtype="bool",
            ),
        }

    def _reset_perception_state(self):
        self.scene_mapper.reset()
        self._is_first_integration = True

    def _get_perception_obs_dict(self, obs, step):
        depth = obs["sensor_data"][self.cam]["depth"]
        rgb = obs["sensor_data"][self.cam]["rgb"]
        segmentation = obs["sensor_data"][self.cam]["segmentation"]
        intrinsics = obs["sensor_param"][self.cam]["intrinsic_cv"]
        extrinsic_cv = obs["sensor_param"][self.cam]["extrinsic_cv"]

        # unify batched (1, H, W, C) shapes
        if depth.ndim == 4:
            depth = depth[0]
            rgb = rgb[0]
            segmentation = segmentation[0]
            intrinsics = intrinsics[0]
            extrinsic_cv = extrinsic_cv[0]

        cam2world_cv = cam2world_cv_from_extrinsic_cv(extrinsic_cv.cpu())

        # SHARED prep+integrate path (same code as the offline dataset converter)
        self.scene_mapper.integrate_raw_frame(
            depth[..., 0], rgb, segmentation[..., 0], intrinsics, cam2world_cv,
            robot_seg_ids=self._robot_seg_ids,
            decay=not self._is_first_integration,
        )
        self._is_first_integration = False

        # The mesh has no fixed size, so it is sampled down here with the same function and
        # sampling method the dataset applies during training.
        vertices, vertex_features = self.scene_mapper.get_scene_representation()
        vertices, vertex_features, vertices_valid_mask = sample_to_n_vertices(
            vertices, vertex_features, self.num_vertices, self.vertex_sampling_method)
        return {
            "vertices": vertices,
            "vertex_features": vertex_features,
            "vertices_valid_mask": vertices_valid_mask,
        }
