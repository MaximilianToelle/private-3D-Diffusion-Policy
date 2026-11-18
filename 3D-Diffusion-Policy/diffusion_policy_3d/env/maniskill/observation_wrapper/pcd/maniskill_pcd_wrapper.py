import torch
from gym import spaces

from diffusion_policy_3d.baseline_scene_integration.perception_utils import (
    backproject_to_world,
    cam2world_cv_from_extrinsic_cv,
    depth_to_meters,
    farthest_point_downsample,
    fused_pcd,
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


class FusedPCDManiSkillDP3Wrapper(ManiSkillDP3BaseObsWrapper):
    """
    Fused point-cloud wrapper (fused_pcd_dense / fused_pcd_sparse).

    Fuses every camera of the current step with perception_utils.fused_pcd, the function
    convert_to_fused_pcd_memmap.py builds the dataset with, and reduces the result to
    ``num_points`` by farthest-point sampling without a random start, as the validation path
    of the PointCloudFPS augmentation does. With point_normals every point carries its unit
    surface normal as three more channels. The random subset fused_pcd draws comes from a
    generator reseeded at every reset, so a start state yields the same subsets in every
    evaluation as long as the rendered images are the same.

    ManiSkill renders nothing inside its step (obs_mode state_dict, asserted by the builder).
    This wrapper renders what a step needs itself: the video camera's rgb at every step, and the
    depth, normals and segmentation of every camera only at the steps whose observation the
    policy reads (observe), the last n_obs_steps of an action chunk. The images stay on the GPU,
    where fused_pcd runs (copying a texture to the CPU takes 5-7 ms, reading it on the GPU 1.3 ms).
    """

    POOL_SEED = 0

    def __init__(self, env, representation_space, agent_proprio_dim, video_cam_name, scene_representation,
                 segmented_body_names, point_normals, device):
        # Perception params must be set BEFORE super().__init__() (which calls
        # _perception_observation_space, reading self.num_points).
        self.scene_representation = scene_representation
        self.num_points = scene_representation["num_points"]
        self.point_normals = point_normals
        self.compute_device = torch.device(device)
        self.generator = torch.Generator(device=self.compute_device)

        super().__init__(env, representation_space, agent_proprio_dim, render_cam_name=video_cam_name)

        # Without a crop box, fused_pcd keeps the points whose segmentation id this marks.
        segmentation_id_map = self.env.unwrapped.segmentation_id_map
        self.is_body = torch.zeros(max(segmentation_id_map) + 1, dtype=torch.bool, device=self.compute_device)
        for seg_id, body in segmentation_id_map.items():
            self.is_body[seg_id] = body.name in segmented_body_names

    def _perception_observation_space(self):
        return {
            'point_cloud': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.num_points, 6 if self.point_normals else 3), dtype='float32'),
        }

    def _reset_perception_state(self):
        self.generator.manual_seed(self.POOL_SEED)

    def _render_cameras(self, observe):
        """sensor_param and sensor_data as ManiSkill's step would return them, rendered in the
        same order (BaseEnv._get_obs_with_sensor_data) and with the same texture transforms
        (Camera.get_obs), but holding only the images this step needs, on the GPU."""
        base_env = self.env.unwrapped
        cameras = sorted(base_env.scene.sensors) if observe else [self.render_cam_name]
        sensor_param = {name: base_env.scene.sensors[name].get_params() for name in cameras}
        for actor in base_env._hidden_objects:
            actor.hide_visual()
        base_env.scene.update_render()
        sensor_data = {}
        for name in cameras:
            camera = base_env.scene.sensors[name]
            camera.capture()
            modalities = ({'depth', 'segmentation', 'normal'} if observe else set()) | ({'rgb'} if name == self.render_cam_name else set())
            shader = camera.config.shader_config
            sensor_data[name] = {}
            for texture_name, texture_modalities in shader.texture_names.items():
                if modalities & set(texture_modalities):
                    # a view of the renderer's buffer, which the next capture overwrites
                    texture = camera.camera._render_cameras[0].get_picture_cuda(texture_name).torch()[None]
                    sensor_data[name].update({modality: image.clone() for modality, image in
                                              shader.texture_transforms[texture_name](texture).items() if modality in modalities})
        return sensor_param, sensor_data

    def _get_obs_dict(self, obs, step, observe=True):
        sensor_param, sensor_data = self._render_cameras(observe)
        obs = dict(obs, sensor_param=sensor_param, sensor_data=sensor_data)
        self._cache_render_frame(obs)
        obs_dict = {'agent_proprio': self._extract_agent_proprio(obs)}
        if observe:
            obs_dict.update(self._get_perception_obs_dict(obs, step))
        return obs_dict

    def _get_perception_obs_dict(self, obs, step):
        # The leading index drops the batch of the single environment, the trailing one the
        # single depth / segmentation channel; cameras in the converter's (sorted) order.
        frames = []
        for camera in sorted(obs['sensor_data']):
            sensor, params = obs['sensor_data'][camera], obs['sensor_param'][camera]
            frames.append((
                depth_to_meters(sensor[self.depth_key][0, ..., 0].to(self.compute_device)),
                sensor['normal'][0].to(self.compute_device).float(),
                sensor['segmentation'][0, ..., 0].to(self.compute_device).long(),
                params['intrinsic_cv'][0].to(self.compute_device),
                cam2world_cv_from_extrinsic_cv(params['extrinsic_cv'][0].to(self.compute_device)),
            ))
        points, normals = fused_pcd(*zip(*frames), self.is_body, self.scene_representation, self.generator)
        cloud = torch.cat([points, normals], dim=-1) if self.point_normals else points
        return {
            'point_cloud': farthest_point_downsample(
                cloud.unsqueeze(0), self.num_points, random_start_point=False).squeeze(0),
        }
