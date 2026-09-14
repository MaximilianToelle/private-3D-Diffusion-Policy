import torch
from gym import spaces

from diffusion_policy_3d.baseline_scene_integration.perception_utils import (
    backproject_to_world,
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


class SingleStepPCDManiSkillDP3Wrapper(ManiSkillDP3BaseObsWrapper):
    """
    Single-step point-cloud baseline wrapper.

    Projects the current camera depth into the sapien world frame, keeps only the robot
    links and movable actors (dropping the workspace table / background via segmentation),
    and downsamples to exactly ``num_points`` via farthest-point sampling. It holds no
    per-episode state, so ``_reset_perception_state`` is left as the base no-op -- which is
    exactly what the spatial-memory wrapper next door adds.

    Depth conversion, backprojection and downsampling all come from perception_utils.py, so
    this wrapper contributes only what is specific to a live ManiSkill env.
    """

    def __init__(self, env, representation_space, agent_proprio_dim, cam_name, num_points):
        # Perception params must be set BEFORE super().__init__() (which calls
        # _perception_observation_space, reading self.num_points).
        self.cam = cam_name
        self.num_points = num_points

        super().__init__(env, representation_space, agent_proprio_dim, render_cam_name=cam_name)

        self.tracked_seg_ids = torch.tensor(
            list(tracked_rigid_bodies_by_seg_id(self.env)), device=self.device,
            dtype=torch.long)

    def _perception_observation_space(self):
        return {
            'point_cloud': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.num_points, 3), dtype='float32'),
        }

    def _get_perception_obs_dict(self, obs, step):
        # ManiSkill batches sensor data over environments and a rollout runs a single one, so the
        # leading index drops that dimension and the trailing one the single depth channel.
        points_world, seg_ids = backproject_to_world(
            depth_to_meters(obs['sensor_data'][self.cam]['depth'][0, ..., 0]),
            obs['sensor_param'][self.cam]['intrinsic_cv'][0],
            cam2world_cv_from_extrinsic_cv(
                obs['sensor_param'][self.cam]['extrinsic_cv'][0]),
            per_pixel_labels=obs['sensor_data'][self.cam]['segmentation'][0, ..., 0],
        )
        tracked_points = points_world[torch.isin(seg_ids, self.tracked_seg_ids)]

        # Deterministic, so that repeated evaluations of one checkpoint see identical
        # observations.
        return {
            'point_cloud': farthest_point_downsample(
                tracked_points.unsqueeze(0), self.num_points,
                random_start_point=False).squeeze(0),
        }
