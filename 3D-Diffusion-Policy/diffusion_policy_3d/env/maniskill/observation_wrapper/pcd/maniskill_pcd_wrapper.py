import torch
from gym import spaces
from torch.nn.utils.rnn import pad_sequence

from diffusion_policy_3d.baseline_scene_integration.perception_utils import (
    backproject_to_world,
    cam2world_cv_from_extrinsic_cv,
    depth_to_meters,
    farthest_point_downsample,
    fused_pcd,
    regional_farthest_point_downsample,
)
from diffusion_policy_3d.baseline_scene_integration.roi_selection import (
    GRASP,
    REGIONS,
    hole_radius,
    next_phase,
    phased_roi_pcd,
    region_budgets,
    region_lookup,
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


class ObservedStepRenderingWrapper(ManiSkillDP3BaseObsWrapper):
    """
    Base of the wrappers that render the cameras themselves, only at the steps whose observation
    the policy reads (observe), the last n_obs_steps of an action chunk, and the video camera's
    rgb at every step. ManiSkill renders nothing inside its step (obs_mode state_dict, asserted by
    the builders). The images stay on the GPU, where the point clouds are built (copying a texture
    to the CPU takes 5-7 ms, reading it on the GPU 1.3 ms). Subclasses build their perception from
    _camera_frames.
    """

    def __init__(self, env, representation_space, agent_proprio_dim, video_cam_name, device):
        self.compute_device = torch.device(device)
        super().__init__(env, representation_space, agent_proprio_dim, render_cam_name=video_cam_name)

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

    def _camera_frames(self, obs):
        """Per camera, in the converter's (sorted) order, on the compute device: depth in meters,
        normal (OpenGL camera frame), segmentation, intrinsic_cv and cam2world_cv, as
        convert_to_fused_pcd_memmap.camera_frames reads them from a recording. The leading index
        drops the batch of the single environment, the trailing one the single depth /
        segmentation channel."""
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
        return list(zip(*frames))


class FusedPCDManiSkillDP3Wrapper(ObservedStepRenderingWrapper):
    """
    Fused point-cloud wrapper (fused_pcd_dense / fused_pcd_sparse).

    Fuses every camera of the current step with perception_utils.fused_pcd, the function
    convert_to_fused_pcd_memmap.py builds the dataset with, and reduces the result to
    ``num_points`` by farthest-point sampling without a random start, as the validation path
    of the PointCloudFPS augmentation does. Without a random start, sampling the dataset's
    farthest-point pool returns the same points as sampling the whole cloud, so the wrapper skips
    the pool (perception_utils.farthest_point_pool). With point_normals every point carries its
    unit surface normal as three more channels.
    """

    def __init__(self, env, representation_space, agent_proprio_dim, video_cam_name, scene_representation,
                 segmented_body_names, point_normals, device):
        # Perception params must be set BEFORE super().__init__() (which calls
        # _perception_observation_space, reading self.num_points).
        self.scene_representation = scene_representation
        self.num_points = scene_representation["num_points"]
        self.point_normals = point_normals

        super().__init__(env, representation_space, agent_proprio_dim, video_cam_name, device)

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

    def _get_perception_obs_dict(self, obs, step):
        points, normals = fused_pcd(*self._camera_frames(obs), self.is_body, self.scene_representation)
        cloud = torch.cat([points, normals], dim=-1) if self.point_normals else points
        return {
            'point_cloud': farthest_point_downsample(
                cloud.unsqueeze(0), self.num_points, random_start_point=False).squeeze(0),
        }


class PhasedROIManiSkillDP3Wrapper(ObservedStepRenderingWrapper):
    """
    Phased region-of-interest wrapper (fused_pcd_phased, baseline_scene_integration/roi_selection.py).

    Builds the per-region clouds of every camera of the current step with
    roi_selection.phased_roi_pcd, the function convert_to_fused_pcd_memmap.py --variant phased
    builds the dataset with, from the env's ground-truth segmentation and hole pose, and samples
    them as the validation path of the RegionalFPS augmentation does: the regions the
    roi_selection mode samples in the current phase, num_points split equally, without a random
    start, which returns the same points as sampling the dataset's per-region farthest-point
    pools. The phase follows the executed gripper commands as in the converter and starts at
    GRASP every reset.
    """

    def __init__(self, env, representation_space, agent_proprio_dim, video_cam_name, scene_representation,
                 roi_selection, point_normals, device):
        # Perception params must be set BEFORE super().__init__() (which calls
        # _perception_observation_space, reading self.num_points).
        self.scene_representation = scene_representation
        self.num_points = scene_representation["num_points"]
        self.point_normals = point_normals

        super().__init__(env, representation_space, agent_proprio_dim, video_cam_name, device)

        base_env = self.env.unwrapped
        self.region_of_segmentation_id = region_lookup(
            {seg_id: body.name for seg_id, body in base_env.segmentation_id_map.items()}).to(self.compute_device)
        self.hole_radius = hole_radius(base_env.piece)
        self.region_budgets = region_budgets(roi_selection, self.num_points).to(self.compute_device)
        self.phase = GRASP

    def _perception_observation_space(self):
        return {
            'point_cloud': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.num_points, 6 if self.point_normals else 3), dtype='float32'),
        }

    def _reset_perception_state(self):
        self.phase = GRASP

    def step(self, action, observe=True):
        # the observation after this action already follows its gripper command, as in the converter
        self.phase = next_phase(self.phase, float(action[-1]))
        return super().step(action, observe)

    def _get_perception_obs_dict(self, obs, step):
        hole_center_xy = self.env.unwrapped.hole_pose.p[0, :2].to(self.compute_device, torch.float32)
        points, normals, regions = phased_roi_pcd(
            *self._camera_frames(obs), self.region_of_segmentation_id, hole_center_xy, self.hole_radius,
            self.scene_representation)
        cloud = torch.cat([points, normals], dim=-1) if self.point_normals else points
        region_clouds = [cloud[regions == region] for region in range(len(REGIONS))]
        num_region_points = torch.tensor([len(region_cloud) for region_cloud in region_clouds], device=cloud.device)
        return {
            'point_cloud': regional_farthest_point_downsample(
                pad_sequence(region_clouds, batch_first=True).unsqueeze(0), num_region_points.unsqueeze(0),
                self.region_budgets[self.phase].unsqueeze(0), random_start_point=False).squeeze(0),
        }
