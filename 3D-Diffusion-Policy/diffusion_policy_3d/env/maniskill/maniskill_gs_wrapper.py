import gym
from gym import spaces
import torch
import time
from functools import partial

import viser
from pytorch3d.transforms import matrix_to_quaternion
from gsworld.mani_skill.utils.gsplat_viewer.gsplat_viewer import GsplatViewer
from gsworld.mani_skill.utils.gsplat_viewer.utils_rasterize_render import _viewer_render_fn, _on_connect
from diffusion_policy_3d.common.transform_utils import quaternion_to_matrix_torch


class GSManiskillDP3Wrapper(gym.Env):
    """
    Wrapper that expects its underlying environment to be a `GSWorldWrapper`.
    - Extracts the 14-channel Gaussian splats from `gs_movable_pts`, 
    - converts quaternions into rotation matrices and 
    - subsamples random gaussians with high opacity for the policy.
    """

    def __init__(
        self, 
        env, 
        num_gaussians=1024, 
        use_gsplat_viewer=False,
    ):
        super().__init__()
        self.env = env
        self.env_got_reset = True
        self.num_gaussians = num_gaussians

        # Cast Gymnasium action space to legacy Gym action space for DP3's wrapper math
        orig_as = self.env.action_space
        self.action_space = spaces.Box(
            low=orig_as.low,
            high=orig_as.high,
            shape=orig_as.shape,
            dtype=orig_as.dtype
        )

        obs, _ = self.env.reset()
        dummy_state = self._extract_state(obs)
        self.obs_sensor_dim = dummy_state.shape[0]

        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.obs_sensor_dim,),
                dtype='float32'
            ),
            'gs_positions': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.num_gaussians, 3), 
                dtype='float32'
            ),
            'gs_rotations_9d': spaces.Box(
                low=-float(1), high=float(1),
                shape=(self.num_gaussians, 9), 
                dtype='float32'
            ),
            'gs_log_scales': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.num_gaussians, 3), 
                dtype='float32'
            ),
            'gs_opacities': spaces.Box(
                low=float(0), high=float(1),
                shape=(self.num_gaussians, 1), 
                dtype='float32'
            ),
            'gs_rgb': spaces.Box(
                low=float(0), high=float(1),
                shape=(self.num_gaussians, 3), 
                dtype='float32'
            ),
        })

        # Init Gsplat Viewer
        self.use_gsplat_viewer = use_gsplat_viewer
        device = self.env.initial_merger_robot._xyz.device
        if self.use_gsplat_viewer:
            self.gs4viewer = {
                "means": torch.zeros((self.num_gaussians, 3), dtype=torch.float32, device=device),
                "quats": torch.zeros((self.num_gaussians, 4), dtype=torch.float32, device=device),
                "scales": torch.zeros((self.num_gaussians, 3), dtype=torch.float32, device=device),
                "rgb_colors": torch.zeros((self.num_gaussians, 3), dtype=torch.float32, device=device),
                "opacities": torch.zeros((self.num_gaussians,), dtype=torch.float32, device=device),
            }
            server = viser.ViserServer(port=8081, verbose=False)
            self.viewer = GsplatViewer(
                        server=server,
                        render_fn=lambda camera_state, render_tab_state: _viewer_render_fn(
                            camera_state, 
                            render_tab_state, 
                            self.gs4viewer, 
                            "3dgs", 
                            device
                        ),
                        output_dir=None,
                        mode="training",
                    )
            scene_center = (self.gs4viewer["means"].data
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
            time.sleep(1)
            server.on_client_connect(partial(_on_connect, server=server, scene_center=scene_center))


    def _extract_state(self, obs):
        # We assume the env returns unbatched shapes or batched arrays of size (1, ...)
        qpos = obs['agent']['qpos']
        if len(qpos.shape) > 1:
            qpos = qpos[0]
        return qpos.float()

    def _get_obs_dict(self, obs, step):
        if self.use_gsplat_viewer:
            while self.viewer.state == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            tic = time.time()

        state = self._extract_state(obs)

        # Base 3DGS Model extraction
        # GSWorldWrapper updates `self.env.gs_movable_pts` internally
        # which is a dict mapping link_name -> (link_xyz, link_scaling, link_rotation, link_opacity)
        # Note: we need the SH DC features too, which are static from `initial_merger_robot`
        
        base_merger = self.env.initial_merger_robot
        
        gsplat_data = {
            "gs_positions": [],        # (-inf, inf)
            "gs_rotations_9d": [],     # [-1, 1]
            "gs_log_scales": [],       # (-inf, inf)
            "gs_opacities": [],        # [0, 1]
            "gs_rgb": [],              # [0, 1]
        }
        for key, value in self.env.gs_movable_pts.items():
            # NOTE: gs_movable_pts does not contain background and workspace table. 
            xyz, log_scales, quats, logit_opacities = value
            
            gsplat_data["gs_positions"].append(xyz.view(-1, 3))
            
            # Converting to 9d rotation matrix for encoding
            quats = torch.nn.functional.normalize(quats, dim=-1)
            rotations_9d = quaternion_to_matrix_torch(quats).reshape(-1, 9)
            gsplat_data["gs_rotations_9d"].append(rotations_9d)

            gsplat_data["gs_log_scales"].append(log_scales.view(-1, 3))

            opacities = torch.sigmoid(logit_opacities)
            gsplat_data["gs_opacities"].append(opacities.view(-1, 1))                      

            # Re-fetch semantic indices to grab features_dc identically
            indices = self.env._semantic_indices[key]
            f_dc = base_merger._features_dc[indices]
            
            if len(f_dc.shape) == 3:
                f_dc = f_dc.squeeze(1)

            SH_C0 = 0.28209479177387814
            # TODO: clamp is not correct if f_dc * SH_C0 + 0.5 outputs values between 0 and 255!! 
            rgb = torch.clamp(f_dc * SH_C0 + 0.5, 0.0, 1.0)
            gsplat_data["gs_rgb"].append(rgb.view(-1, 3))

        for key, value in gsplat_data.items():
            merged_value = torch.cat(value, dim=0)
            gsplat_data[key] = merged_value

        # === Transform Gaussians from GS robot arm coordinates to SAPIEN base coordinates ===
        # NOTE: Actors have been reconstructed in their own coordinate system. Inside gs_world_wrapper, they get mapped into GS robot arm coordinates. Therefore, gs_movable_pts contains all Gaussians wrt to that coordinate frame 
        
        # Get the global transform and invert it to map points back to SAPIEN frame
        sim2gs_trans = self.env.sim2gs_arm_trans 
        gs2sim_trans = torch.linalg.inv(sim2gs_trans) # Shape: (4, 4)

        # Extract rotation, translation and scale
        R_unnorm = gs2sim_trans[:3, :3]
        t_gs2sim = gs2sim_trans[:3, 3]

        # NOTE: working with uniform scale assumption!
        scales = torch.linalg.norm(R_unnorm, dim=0)
        assert torch.allclose(scales, scales[0], atol=1e-5), f"Assumed uniform scale, but axis scales differ: {scales}"
        scale_gs2sim = scales[0]
        R_gs2sim = R_unnorm / scale_gs2sim

        # 1. Transform XYZ positions
        xyz_gs = gsplat_data["gs_positions"]
        gsplat_data["gs_positions"] = xyz_gs @ R_unnorm.T + t_gs2sim

        # 2. Transform Rotation Matrices
        rot_mats_gs = gsplat_data["gs_rotations_9d"].reshape(-1, 3, 3)
        rot_mats_sim = R_gs2sim.unsqueeze(0) @ rot_mats_gs
        gsplat_data["gs_rotations_9d"] = rot_mats_sim.reshape(-1, 9)

        # 3. Transform lengths (scaling parameters)
        log_scales_gs = gsplat_data["gs_log_scales"]
        gsplat_data["gs_log_scales"] = log_scales_gs + torch.log(scale_gs2sim)

        # === DEBUG VISUALIZATION ===
        #import open3d as o3d
        #import numpy as np

        # 1. Get centers
        #centers = new_xyz.detach().cpu().numpy()
        # 2. Get colors from features_dc (SH0 to RGB approx)
        #sh0 = merged_gaussians[:, 16:19].detach().cpu()  # f_dc is at indices 16, 17, 18
        #colors = torch.clamp(sh0 * 0.28209 + 0.5, 0.0, 1.0).numpy()  # open3d expects [0, 1] floats

        # 3. Create PointCloud
        #pc = o3d.geometry.PointCloud()
        #pc.points = o3d.utility.Vector3dVector(centers)
        #pc.colors = o3d.utility.Vector3dVector(colors)

        # 4. Create SAPIEN Base Coordinate Frame
        # Since we mapped everything back to the SAPIEN base,
        # the base is exactly at the origin (0, 0, 0)
        # Red=X, Green=Y, Blue=Z
        #axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

        # 5. Visualize interactively
        #print(f"Opening interactive visualization with {len(centers)} Gaussians... (Close window to continue)")
        #o3d.visualization.draw_geometries([pc, axis], window_name="Debug Gaussians")

        if self.env_got_reset: 
            # Random sampling of high opacity Gaussians for new episode
            pts = gsplat_data["gs_positions"]
            opacities = gsplat_data["gs_opacities"]

            high_opacity_mask = (opacities >= 0.98) # needs to be same as in dataset conversion!
            valid_indices = torch.where(high_opacity_mask)[0]
            assert (len(valid_indices) >= self.num_gaussians), "Less high opacity Gaussians than requested for subsampling!"

            pts_valid = pts[valid_indices]
            N_total = pts_valid.shape[0]
            sampled_indices = torch.randperm(N_total)[:self.num_gaussians]
            
            # Map back to original indices so we can index original pts and feats later
            self.gaussian_indices = valid_indices[sampled_indices.squeeze(0)]  # (npoint,)

        for key, value in gsplat_data.items():
            subsampled_values = value[self.gaussian_indices]
            gsplat_data[key] = subsampled_values

        # Save RGB for rendering
        if 'sensor_data' in obs and 'right_cam' in obs['sensor_data']:
            rgb = obs['sensor_data']['right_cam']['rgb'].squeeze(0)
            self._last_rgb = rgb.clone().to(torch.uint8)

        # Visualize in GsplatViewer
        # Update gaussians for gsplatviewer 
        if self.use_gsplat_viewer:
            for key in gsplat_data:
                if key == "gs_positions": 
                    self.gs4viewer["means"] = gsplat_data[key]
                elif key == "gs_rotations_9d":
                    N = gsplat_data["gs_rotations_9d"].shape[0]
                    rot_matrices = gsplat_data["gs_rotations_9d"].reshape(N, 3, 3)
                    self.gs4viewer["quats"] = matrix_to_quaternion(rot_matrices)
                elif key == "gs_log_scales":
                    self.gs4viewer["scales"] = gsplat_data[key]
                elif key == "gs_rgb":
                    self.gs4viewer["rgb_colors"] = gsplat_data[key]
                elif key == "gs_opacities": 
                    logit_viewer_opacities = torch.logit(gsplat_data["gs_opacities"])
                    self.gs4viewer["opacities"] = logit_viewer_opacities.view(-1)

            self.viewer.rerender(None)
            self.viewer.lock.release()
            num_train_rays_per_step = (3 * 540 * 960)
            num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
            num_train_rays_per_sec = (num_train_rays_per_step * num_train_steps_per_sec)
            self.viewer.render_tab_state.num_train_rays_per_sec = (num_train_rays_per_sec)
            self.viewer.update(step, num_train_rays_per_step)  

        obs_dict = gsplat_data
        obs_dict["agent_pos"] = state

        return obs_dict

    def step(self, action):
        if hasattr(self.env.unwrapped, "num_envs") and self.env.unwrapped.num_envs > 0:
            if not isinstance(action, torch.Tensor):
                action = torch.from_numpy(action)
            action = action.unsqueeze(0)

        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_dict = self._get_obs_dict(obs, info['elapsed_steps'].item())

        if hasattr(terminated, "item"):
            terminated = bool(terminated.item())
        if hasattr(truncated, "item"):
            truncated = bool(truncated.item())
        if hasattr(reward, "item"):
            reward = float(reward.item())

        if isinstance(terminated, (torch.Tensor,)) and terminated.ndim > 0:
            terminated = bool(terminated[0].item())
        if isinstance(truncated, (torch.Tensor,)) and truncated.ndim > 0:
            truncated = bool(truncated[0].item())
        if isinstance(reward, (torch.Tensor,)) and reward.ndim > 0:
            reward = float(reward[0].item())

        done = bool(terminated or truncated)
        reward = float(reward)
        return obs_dict, reward, done, info

    def reset(self, **kwargs):
        # Reset sampled gaussian indices for new episode
        obs, info = self.env.reset(**kwargs)

        self.gaussian_indices = None
        self.env_got_reset = True

        obs_dict = self._get_obs_dict(obs, info['elapsed_steps'].item())

        self.env_got_reset = False
        return obs_dict

    def render(self, mode="rgb_array"):
        if hasattr(self, '_last_rgb'):
            return self._last_rgb.cpu().numpy()
        return torch.zeros((256, 256, 3), dtype=torch.uint8).numpy()