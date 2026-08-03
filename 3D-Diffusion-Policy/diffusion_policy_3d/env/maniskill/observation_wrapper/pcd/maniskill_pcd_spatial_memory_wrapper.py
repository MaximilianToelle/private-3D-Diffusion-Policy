"""Live-environment half of the spatial_memory_pcd baseline.

A *scene mapper* is the stateful reconstruction component of a baseline: it consumes one
preprocessed camera frame at a time and maintains the scene representation the policy
consumes, under the contract of baseline_scene_integration/base_scene_mapper.py. This wrapper
receives one instead of constructing it, because the identical mapper also runs offline in
scripts/dataset/conversion/convert_wrist_cam_gsworld_to_spatial_memory_pcd.py, which is not an
environment. The online instance comes from env_runner/maniskill_env_obs_wrapper_builder.py,
in spatial_memory_pcd -- see the architecture section of the repository README for the pattern
and for the config-versus-dataset assert that construction site carries.
"""

import torch
from gym import spaces

from diffusion_policy_3d.baseline_scene_integration.perception_utils import (
    cam2world_cv_from_extrinsic_cv,
    depth_to_meters,
    farthest_point_downsample,
)
from diffusion_policy_3d.env.maniskill.observation_wrapper.maniskill_dp3_base_obs_wrapper import (
    ManiSkillDP3BaseObsWrapper,
)
from diffusion_policy_3d.env.maniskill.observation_wrapper.maniskill_scene_infos import (
    tracked_rigid_bodies_by_seg_id,
)


class SpatialMemoryPCDManiSkillDP3Wrapper(ManiSkillDP3BaseObsWrapper):
    """Observation wrapper of the spatial-memory baseline.

    Rather than seeing only the current step's depth-projected points, the policy receives
    a point cloud accumulated over the whole episode, written into
    ``obs_dict['point_cloud']`` at every step. The accumulation itself, meaning
    per-rigid-body re-posing and voxel dedup, lives in ``SpatialMemoryPcdSceneMapper``, and
    the fixed-size cloud is drawn from that memory by the same ``farthest_point_downsample``
    that the PointCloudFPS augmentation applies during training.

    What stays here is everything SAPIEN-specific, which the mapper must not know about
    because its offline twin is fed GS semantic ids and FK / env_states poses instead:

      * the lookup from segmentation id to tracked-rigid-body index, where the robot links
        and the movable actors have an index while table, ground and background map to -1, and
      * the current (K, 4, 4) pose of every tracked rigid body, read once per step off the
        live ``Link`` and ``Actor`` structs.
    """

    def __init__(self, env, representation_space, agent_proprio_dim, cam_name, scene_mapper,
                 num_points):
        # Perception params must be set BEFORE super().__init__() (which calls
        # _perception_observation_space, reading self.num_points).
        self.cam = cam_name
        self.num_points = num_points

        super().__init__(env, representation_space, agent_proprio_dim, render_cam_name=cam_name)
        self.scene_mapper = scene_mapper

        # The mapper refers to a rigid body by its index in a list whose order must stay fixed
        # for the whole episode, so that list and a dense segmentation-id lookup into it are
        # built once here. Both feed every integrate_frame call: the lookup turns per-pixel
        # segmentation ids into body indices in a single gather, the list yields the poses.
        tracked_rigid_body_by_seg_id = tracked_rigid_bodies_by_seg_id(self.env)
        self._tracked_rigid_bodies = list(tracked_rigid_body_by_seg_id.values())
        tracked_seg_ids = torch.tensor(list(tracked_rigid_body_by_seg_id),
                                       device=self.device, dtype=torch.long)
        max_seg_id = max(self.env.unwrapped.segmentation_id_map)
        self._seg_id_to_rigid_body_index = torch.full(
            (int(max_seg_id) + 1,), -1, dtype=torch.long, device=self.device)
        self._seg_id_to_rigid_body_index[tracked_seg_ids] = torch.arange(
            tracked_seg_ids.numel(), device=self.device)

    def _perception_observation_space(self):
        return {
            'point_cloud': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.num_points, 3), dtype='float32'),
        }

    def _current_rigid_body_to_world_poses(self):
        """The (K, 4, 4) body-to-world transform of every tracked rigid body, in the order the
        mapper's rigid-body indices refer to. ManiSkill poses are always batched over
        environments, and a rollout runs a single one."""
        return torch.stack([
            rigid_body.pose.to_transformation_matrix()[0].to(self.device)
            for rigid_body in self._tracked_rigid_bodies])

    def _get_perception_obs_dict(self, obs, step):
        # ManiSkill batches sensor data over environments and a rollout runs a single one, so the
        # leading index drops that dimension and the trailing one the single depth channel. The
        # env records int16 millimeters, hence the shared conversion.
        depth_m = depth_to_meters(obs["sensor_data"][self.cam]["depth"][0, ..., 0])
        segmentation = obs["sensor_data"][self.cam]["segmentation"][0, ..., 0]
        intrinsics = obs["sensor_param"][self.cam]["intrinsic_cv"][0]
        cam2world_cv = cam2world_cv_from_extrinsic_cv(
            obs["sensor_param"][self.cam]["extrinsic_cv"][0])

        # These poses are needed twice: for the inverse transform that stores fresh points
        # in the frame of their rigid body, and for the forward transform that renders the
        # memory back into the world frame.
        rigid_body_to_world_poses = self._current_rigid_body_to_world_poses()

        self.scene_mapper.integrate_frame(
            depth_m,
            segmentation,
            intrinsics,
            cam2world_cv,
            self._seg_id_to_rigid_body_index,
            rigid_body_to_world_poses,
        )

        # The memory itself has no fixed size, so it is sampled down here. The cloud is exactly
        # as long as the memory, hence no lengths, and the sampling is deterministic so that
        # repeated evaluations of one checkpoint on one seed see identical observations.
        accumulated_cloud = self.scene_mapper.get_scene_representation(rigid_body_to_world_poses)
        return {
            'point_cloud': farthest_point_downsample(
                accumulated_cloud.unsqueeze(0), self.num_points,
                random_start_point=False).squeeze(0),
        }

    def _reset_perception_state(self):
        # Clear the spatial memory at the start of every episode. The base wrapper calls
        # this before the first _get_obs_dict(), so the memory is empty when the initial
        # observation is built.
        self.scene_mapper.reset()
