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
    and reduces them to exactly ``num_points`` by one deterministic farthest-point sampling.
    That equals the dataset's deterministic path, the stored ``farthest_point_pool`` of
    convert_to_single_step_pcd_memmap.py sampled down by PointCloudFPS without a random start:
    both greedy selections begin at the same first point, so the pool's first ``num_points``
    picks are exactly the direct picks, and sampling the pool reproduces them (verified
    bit-identical on recorded frames). Skipping the pool stage online saves the sampling of
    ~8k out of ~100k points per step, which is an order of magnitude more expensive than the
    direct sampling. It holds no per-episode state, so ``_reset_perception_state`` is left as
    the base no-op -- which is exactly what the spatial-memory wrapper next door adds.

    Depth conversion, backprojection and downsampling all come from perception_utils.py, so
    this wrapper contributes only what is specific to a live ManiSkill env.
    """

    def __init__(self, env, representation_space, agent_proprio_dim, cam_name, num_points, device):
        # Perception params must be set BEFORE super().__init__() (which calls
        # _perception_observation_space, reading self.num_points).
        self.cam = cam_name
        self.num_points = num_points
        # Backprojection and sampling run here rather than on the simulation's device, since a
        # frame holds ~100k tracked points and farthest-point sampling them takes seconds on the CPU.
        self.compute_device = torch.device(device)

        super().__init__(env, representation_space, agent_proprio_dim, render_cam_name=cam_name)

        self.tracked_seg_ids = torch.tensor(
            list(tracked_rigid_bodies_by_seg_id(self.env)), device=self.compute_device,
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
        sensor, params = obs['sensor_data'][self.cam], obs['sensor_param'][self.cam]
        points_world, seg_ids = backproject_to_world(
            depth_to_meters(sensor[self.depth_key][0, ..., 0].to(self.compute_device)),
            params['intrinsic_cv'][0].to(self.compute_device),
            cam2world_cv_from_extrinsic_cv(params['extrinsic_cv'][0].to(self.compute_device)),
            per_pixel_labels=sensor['segmentation'][0, ..., 0].to(self.compute_device),
        )
        tracked_points = points_world[torch.isin(seg_ids, self.tracked_seg_ids)]

        # Deterministic, so that repeated evaluations of one checkpoint see identical observations.
        return {
            'point_cloud': farthest_point_downsample(
                tracked_points.unsqueeze(0), self.num_points, random_start_point=False).squeeze(0),
        }
