import gym
from gym import spaces
import torch
import numpy as np
import copy
from typing import Optional
from mani_skill.utils.structs.link import Link

# DynaGSLAM internals
from dynagslam.scene.cameras import Camera
from dynagslam.SLAM.multiprocess.mapper_dyna_eval_sam_keti import Mapping
from dynagslam.SLAM.multiprocess.tracker import Tracker
from dynagslam.SLAM.utils import move_to_gpu, move_to_cpu
from dynagslam.utils.graphics_utils import fov2focal


SH_C0 = 0.28209479177387814  # zeroth SH band coefficient


class DynaGSLAMWrapper(gym.Env):
    """
    Drop-in replacement for GSWorldWrapper + GSManiskillDP3Wrapper combined.

    Drives online Gaussian Splatting via DynaGSLAM's Mapping class.
    Rather than using a pre-baked static 3DGS model deformed by kinematics,
    this wrapper reconstructs the Gaussian scene incrementally from live
    RGBD observations, using ground-truth simulation data in place of the
    three costly perception modules DynaGSLAM normally requires:
        • Segmentation  → sim segmentation mask via _get_sim_segmentation()
        • Localization  → sim camera extrinsics (args.use_gt_pose = True)
        • Optical flow  → set to None; DynaGSLAM falls back to pose-only path

    Depth convention:
        ManiSkill returns depth in metres (float32).
        DynaGSLAM's map_preprocess() does `depth_map = frame.original_depth * 255`,
        so we store `depth_metres / 255` in Camera.original_depth to restore metres
        inside the SLAM pipeline. Ensure min_depth / max_depth in slam_args are
        set in metres (e.g. 0.3 … 5.0) to match.

    Wrapper stack:
        gymnasium.make(task_name)
            └── DynaGSLAMWrapper        ← this file
                └── SimpleVideoRecordingWrapper
                    └── MultiStepWrapper

    Output observation dict (identical keys to GSManiskillDP3Wrapper):
        gs_positions:    (num_gaussians, 3)   float32
        gs_rotations_9d: (num_gaussians, 9)   float32  flattened 3×3 rot matrix
        gs_log_scales:   (num_gaussians, 3)   float32
        gs_opacities:    (num_gaussians, 1)   float32  in [0, 1]
        gs_rgb:          (num_gaussians, 3)   float32  in [0, 1]
        agent_pos:       (num_joints,)         float32
    """

    # Segmentation IDs that correspond to dynamic objects.
    # Pixels with these IDs will be treated as foreground by DynaGSLAM.
    # Override / extend in a subclass for task-specific semantics.
    DYNAMIC_SEG_IDS: tuple = ()   # filled by _get_sim_segmentation at runtime

    def __init__(
        self,
        env,
        slam_args,
        optimization_params,
        cam_name: str = "right_cam",
        num_gaussians: int = 1024,
        use_gsplat_viewer: bool = False,
    ):
        """
        Parameters
        ----------
        env : gymnasium.Env
            Raw ManiSkill environment (output of gymnasium.make).
        slam_args : Namespace
            Parsed DynaGSLAM config. Must have use_gt_pose=True, mode='single process'.
        optimization_params : OptimizationParams
            DynaGSLAM optimization hyper-parameters (lr, weights, …).
        cam_name : str
            Name of the ManiSkill camera whose RGBD feed is given to SLAM.
        num_gaussians : int
            Number of Gaussians to subsample for the policy observation.
        use_gsplat_viewer : bool
            If True, launch an interactive viser viewer showing the live GS map.
        """
        super().__init__()
        self.env = env
        self.cam_name = cam_name
        self.num_gaussians = num_gaussians
        self.use_gsplat_viewer = use_gsplat_viewer
        self.slam_args = slam_args
        self.optimization_params = optimization_params

        # Force sim-compatible DynaGSLAM settings
        self.slam_args.use_gt_pose = True
        self.slam_args.mode = "single process"

        self._frame_id: int = 0
        self._gaussian_indices: Optional[torch.Tensor] = None
        self._env_got_reset: bool = False
        self._last_rgb: Optional[np.ndarray] = None

        # SLAM objects — created fresh on each reset()
        self.gaussian_map: Optional[Mapping] = None
        self._tracker_preprocessor: Optional[Tracker] = None

        # Cast Gymnasium action space → legacy Gym for MultiStepWrapper
        orig_as = self.env.action_space
        self.action_space = spaces.Box(
            low=orig_as.low, high=orig_as.high,
            shape=orig_as.shape, dtype=orig_as.dtype,
        )

        # Observation space — populated after the first env.reset() so we know
        # agent_pos dimensionality from qpos.
        self.obs_sensor_dim: Optional[int] = None
        self.observation_space: Optional[spaces.Dict] = None
        agent_pos_dim = self.env.observation_space['agent']['qpos'].shape[-1]
        self._build_observation_space(agent_pos_dim)

        self._last_gs_rgb = None

        if use_gsplat_viewer:
            self._init_gsplat_viewer()

    # ------------------------------------------------------------------
    # DynaGSLAM lifecycle
    # ------------------------------------------------------------------

    def _init_slam(self):
        """Instantiate a fresh Mapping + a lightweight Tracker (geometry only)."""
        self.gaussian_map = Mapping(self.slam_args)
        # Tracker is used only for map_preprocess() (normals, vertex maps).
        # No pose estimation is performed because use_gt_pose=True.
        self._tracker_preprocessor = Tracker(self.slam_args)
        self._frame_id = 0

    def _make_dyna_camera(self, obs: dict, frame_id: int) -> Camera:
        """
        Convert a ManiSkill observation dict into a DynaGSLAM Camera object.

        Camera convention (inherited from 3DGS):
            R  = w2c rotation transposed = c2w rotation   (numpy, float64)
            T  = w2c translation                            (numpy, float64)
            pose_gt = c2w 4×4 matrix                       (numpy, float64)

        Depth is stored as depth_metres / 255 so that map_preprocess()'s
        `* 255` restores metric metres.
        """
        cam_params = obs['sensor_param'][self.cam_name]
        cam_data   = obs['sensor_data'][self.cam_name]

        # --- Intrinsics ---
        K = cam_params['intrinsic_cv'][0].float()   # (3, 3) on GPU
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx_px,  cy_px  = float(K[0, 2]), float(K[1, 2])

        rgb = cam_data['rgb']   # (B, H, W, 3) uint8
        H = rgb.shape[1]
        W = rgb.shape[2]

        FoVx = float(2 * torch.arctan(torch.tensor(W / (2 * fx))))
        FoVy = float(2 * torch.arctan(torch.tensor(H / (2 * fy))))

        # --- Pose (ground truth from sim) ---
        # extrinsic_cv: (B, 3, 4) world-to-cam in OpenCV convention
        extr = cam_params['extrinsic_cv'][0].float()   # (3, 4)
        w2c_4x4 = torch.eye(4, dtype=torch.float32)
        w2c_4x4[:3, :] = extr
        c2w_4x4 = torch.linalg.inv(w2c_4x4)

        R = c2w_4x4[:3, :3].cpu().numpy().astype(np.float64)   # c2w rotation = w2c.T
        T = w2c_4x4[:3,  3].cpu().numpy().astype(np.float64)   # w2c translation
        pose_gt = c2w_4x4.cpu().numpy().astype(np.float64)

        # --- RGB tensor (3, H, W) float [0, 1] ---
        rgb_chw = rgb[0].float().permute(2, 0, 1) / 255.0   # (3, H, W)

        # --- Depth tensor (1, H, W) float, stored as metres / 255 ---
        depth_m = cam_data['depth'][0].float()               # (H, W, 1) or (H, W)
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]                        # (H, W)
        # ManiSkill depth is in millimetres; DynaGSLAM expects metres.
        depth_m = depth_m / 1000.0
        depth_chw = (depth_m / 255.0).unsqueeze(0)           # (1, H, W)

        timestamp = frame_id / 20.0    # control_freq = 20 Hz → seconds

        valid_depth = depth_m[torch.isfinite(depth_m) & (depth_m > 0)]
        print(
            "raw depth:",
            depth_m.dtype,
            tuple(depth_m.shape),
            valid_depth.min().item() if valid_depth.numel() else None,
            valid_depth.max().item() if valid_depth.numel() else None,
)

        cam = Camera(
            colmap_id=frame_id,
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            image=rgb_chw,
            depth=depth_chw,
            gt_alpha_mask=None,
            image_name=f"frame_{frame_id:06d}",
            uid=frame_id,
            pose_gt=pose_gt,
            cx=cx_px,
            cy=cy_px,
            timestamp=timestamp,
            depth_scale=1.0,
            preload=True,
            data_device=str(self.slam_args.data_device),
        )
        return cam

    def _get_sim_segmentation(self, obs: dict) -> np.ndarray:
        """
        Build a binary dynamic-object mask from the sim segmentation image.

        Pixels belonging to the robot arm or any manipulated actor are
        marked as dynamic (1); background / static environment = 0.

        DynaGSLAM expects a (H, W) bool / uint8 numpy array.
        The mask is also used to decide which Gaussians to treat as
        dynamic inside Mapping.mapping().

        Strategy: we flag every pixel whose segmentation ID is > 0
        (SAPIEN assigns 0 to the background table / floor and positive
        IDs to actors and robot links). Tune this for your specific task.
        """
        ##seg = obs['sensor_data'][self.cam_name]['segmentation']  # (B, H, W, 1) int16 / int32
        ##seg_2d = seg[0, :, :, 0].cpu().numpy()                  # (H, W)
        ##dynamic_mask = (seg_2d > 0).astype(np.uint8)
        ##return dynamic_mask
        ##print(f"movable_ids: {movable_ids}, unique seg IDs in frame: {np.unique(seg_2d)}")
        seg = obs['sensor_data'][self.cam_name]['segmentation']       # (B, H, W, 1)
        seg_2d = seg[0, :, :, 0].cpu().numpy()                          # (H, W)

        id_map = self.env.unwrapped.segmentation_id_map
        movable_ids = {
        obj_id for obj_id, obj in id_map.items()
        if isinstance(obj, Link) or getattr(obj, "px_body_type", None) == "dynamic"
        }

        dynamic_mask = np.isin(seg_2d, list(movable_ids)).astype(np.uint8)
        return dynamic_mask

    def _run_slam_step(self, obs: dict):
        """
        Execute one full DynaGSLAM mapping step for the current observation.

        Corresponds to one iteration of the inner loop in slam_eval.main():
            1. Build DynaGSLAM Camera from sim obs
            2. Build binary segmentation mask (replaces SAM)
            3. Preprocess frame geometry via Tracker.map_preprocess()
            4. Update camera pose in the frame (use_gt_pose path)
            5. Compute world-space vertex / normal maps
            6. Run Mapping.mapping() — adds / optimises Gaussians
            7. Increment _frame_id
        """
        frame = self._make_dyna_camera(obs, self._frame_id)
        seg_mask = self._get_sim_segmentation(obs)

        move_to_gpu(frame)

        # Geometry preprocessing (vertex map, normal map, confidence map)
        frame_map = self._tracker_preprocessor.map_preprocess(frame, self._frame_id)

        # With use_gt_pose=True, tracking() just copies pose_gt into frame
        # and computes world-space vertex / normal maps — no ICP / ORB needed.
        self._tracker_preprocessor.tracking(frame, frame_map, seg_mask, None)

        # tracking() populates the world-space geometry entries.
        d = frame_map["depth_map"]
        v = frame_map["vertex_map_w"]

        # Pass world-space vertex map back into frame_map so Mapping can use it
        # (tracking() populates frame_map["vertex_map_w"] and "normal_map_w").

        # Run Gaussian map update (add points, local optimise, prune)
        # flow_gt=None → DynaGSLAM falls back to pose-only dynamic association
        # timestamp_curr / timestamp_old based on frame_id counter

        print(
            "processed depth:",
            d.min().item(),
            d.max().item(),
            "nonzero:", (d > 0).sum().item(),
            "/", d.numel(),
        )
        print(
            "vertex finite:", torch.isfinite(v).all().item(),
            "nonzero vertices:", (v.abs().sum(dim=-1) > 0).sum().item(),
        )

        t_curr = frame.timestamp
        t_old  = (self._frame_id - 1) / 20.0 if self._frame_id > 0 else 0.0

        self.gaussian_map.mapping(
            frame,
            frame,               # frame_eval == frame (no separate eval camera in sim)
            frame_map,
            self._frame_id,
            self.optimization_params,
            dyna_mask=seg_mask,
            dyna_mask_eval=seg_mask,
            flow_gt=None,
            t_curr=t_curr,
            t_past=t_old,
        )

        with torch.no_grad():
            render_output = self.gaussian_map.renderer.render(
                frame,
                self.gaussian_map.global_params,
            )

            gs_rgb = render_output["render"]          # (3, H, W), float [0,1]
            gs_rgb = (
                gs_rgb
                .clamp(0.0, 1.0)
                .permute(1, 2, 0)
                .mul(255)
                .byte()
                .cpu()
                .numpy()
            )

            self._last_gs_rgb = gs_rgb


        self.gaussian_map.time += 1
        move_to_cpu(frame)
        self._frame_id += 1

    # ------------------------------------------------------------------
    # Gaussian extraction → policy observation
    # ------------------------------------------------------------------

    def _extract_gaussians_from_map(self) -> dict:
        """
        Read self.gaussian_map.global_params and reformat into policy obs keys.

        global_params keys (from Mapping):
            xyz:       (N, L, 3) positions; renderer uses level 0
            opacity:   (N, 1)    activated opacities
            scales:    (N, 3)    activated positive scales
            rotations: (N, L, 4) wxyz quaternions; renderer uses level 0
            shs:       (N, ?, 3) spherical harmonics; index [0] = DC band

        Returns dict with policy-facing keys.
        """
        gp = self.gaussian_map.global_params   # dict of tensors, all on GPU

        N = gp['xyz'].shape[0]
        if N == 0:
            # Map not yet seeded — return zeros as a safe fallback
            device = torch.device('cuda')
            return {
                'gs_positions':    torch.zeros(1, 3, device=device),
                'gs_rotations_9d': torch.zeros(1, 9, device=device),
                'gs_log_scales':   torch.zeros(1, 3, device=device),
                'gs_opacities':    torch.zeros(1, 1, device=device),
                'gs_rgb':          torch.zeros(1, 3, device=device),
            }

        # DynaGSLAM stores L geometry levels for xyz/rotation. Its renderer uses
        # level 0, so expose that same component to the policy.
        xyz = gp['xyz'][:, 0, :] if gp['xyz'].ndim == 3 else gp['xyz']
        quats = (
            gp['rotations'][:, 0, :]
            if gp['rotations'].ndim == 3
            else gp['rotations']
        )

        # Mapping.global_params returns activated values. Convert scales back to
        # the log representation expected by the policy, but do not sigmoid the
        # already-activated opacity a second time.
        log_scales = torch.log(gp['scales'].clamp_min(1e-12))
        opacities = gp['opacity'].clamp(0.0, 1.0)
        shs = gp['shs']

        # Quaternion (wxyz) → 3×3 rotation matrix → flatten to 9-D
        quats_norm = torch.nn.functional.normalize(quats, dim=-1)
        rot_mats   = _quaternion_wxyz_to_matrix(quats_norm)   # (N, 3, 3)
        rot_9d     = rot_mats.reshape(N, 9)

        # DC SH band → linear RGB approximation
        dc = shs[:, 0, :]                          # (N, 3)
        rgb = torch.clamp(dc * SH_C0 + 0.5, 0.0, 1.0)

        return {
            'gs_positions':    xyz,
            'gs_rotations_9d': rot_9d,
            'gs_log_scales':   log_scales,
            'gs_opacities':    opacities,
            'gs_rgb':          rgb,
        }

    def _subsample_gaussians(self, gsplat_data: dict, force_resample: bool) -> dict:
        """
        Subsample to self.num_gaussians Gaussians.

        Policy:
          • On force_resample (episode reset): filter to opacity >= 0.98 and
            randomly draw num_gaussians. Store self._gaussian_indices.
          • Subsequent steps: reuse the same indices for episode consistency.
            Because the SLAM map grows and Gaussians can be deleted, we clamp
            indices to the current map size and re-sample any out-of-range ones.

        If fewer than num_gaussians high-opacity Gaussians are available
        (map not yet built up), we sample uniformly from all available ones.
        """
        N = gsplat_data['gs_positions'].shape[0]
        device = gsplat_data['gs_positions'].device

        if force_resample or self._gaussian_indices is None:
            opacities = gsplat_data['gs_opacities'].squeeze(-1)  # (N,)
            high_mask = opacities >= 0.98
            valid_idx = torch.where(high_mask)[0]

            if len(valid_idx) >= self.num_gaussians:
                chosen = valid_idx[torch.randperm(len(valid_idx))[:self.num_gaussians]]
            else:
                # Fall back to sampling from all Gaussians
                chosen = torch.randperm(N, device=device)[:min(self.num_gaussians, N)]

            self._gaussian_indices = chosen

        # Clamp stale indices that exceed current map size (map can shrink due to pruning)
        valid = self._gaussian_indices < N
        if not valid.all():
            # Replace invalid indices with fresh random ones
            replacement = torch.randperm(N, device=device)[:int((~valid).sum())]
            self._gaussian_indices[~valid] = replacement

        # Ensure we always have exactly num_gaussians entries by padding if needed
        if len(self._gaussian_indices) < self.num_gaussians:
            extra = self.num_gaussians - len(self._gaussian_indices)
            pad = torch.randperm(N, device=device)[:extra]
            self._gaussian_indices = torch.cat([self._gaussian_indices, pad])

        idx = self._gaussian_indices[:self.num_gaussians]
        return {k: v[idx] for k, v in gsplat_data.items()}

    def _build_obs_dict(self, raw_obs: dict, force_resample: bool = False) -> dict:
        """
        Compose the final policy-facing observation dict.
        """
        # Complete DynaGSLAM map.
        full_gsplat_data = self._extract_gaussians_from_map()

        # Show the complete reconstruction in Viser.
        if self.use_gsplat_viewer:
            self._update_gsplat_viewer(full_gsplat_data)

        # Keep the policy input at num_gaussians, e.g. 1024.
        gsplat_data = self._subsample_gaussians(
            full_gsplat_data,
            force_resample,
        )

        # Agent state: robot joint positions
        qpos = raw_obs['agent']['qpos']
        if qpos.ndim > 1:
            qpos = qpos[0]
        agent_pos = qpos.float()

        # Cache RGB for render()
        rgb_hw3 = raw_obs['sensor_data'][self.cam_name]['rgb'][0]   # (H, W, 3) uint8
        self._last_rgb = rgb_hw3.cpu().numpy()

        obs_dict = {**gsplat_data, 'agent_proprio': agent_pos}


        return obs_dict

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def _build_observation_space(self, agent_pos_dim: int):
        self.obs_sensor_dim = agent_pos_dim # agent_proprio
        self.observation_space = spaces.Dict({
            'agent_proprio': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(agent_pos_dim,), dtype='float32',
            ),
            'gs_positions': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.num_gaussians, 3), dtype='float32',
            ),
            'gs_rotations_9d': spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.num_gaussians, 9), dtype='float32',
            ),
            'gs_log_scales': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.num_gaussians, 3), dtype='float32',
            ),
            'gs_opacities': spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_gaussians, 1), dtype='float32',
            ),
            'gs_rgb': spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_gaussians, 3), dtype='float32',
            ),
        })

    def reset(self, **kwargs):
        """
        Reset the sim, re-initialise SLAM from scratch, seed the map with
        the first frame, and return the initial policy observation dict.
        """
        raw_obs, info = self.env.reset(**kwargs)

        # Fresh SLAM map for every episode
        self._init_slam()
        self._gaussian_indices = None
        self._env_got_reset = True

        # Seed the map with the first frame (no Gaussians yet → map_preprocess
        # + first mapping call initialises the point cloud from depth)
        self._run_slam_step(raw_obs)

        obs_dict = self._build_obs_dict(raw_obs, force_resample=True)
        self._env_got_reset = False

        # Build observation space on first call (agent_pos dim is now known)
        if self.observation_space is None:
            self._build_observation_space(obs_dict['agent_proprio'].shape[0])

        # Return only obs (legacy Gym style) — MultiStepWrapper expects this
        return obs_dict

    def step(self, action):
        """
        Step the underlying sim, update the SLAM map, extract Gaussians.
        """
        # Unsqueeze for batched SAPIEN envs (num_envs dimension)
        if hasattr(self.env.unwrapped, 'num_envs') and self.env.unwrapped.num_envs > 0:
            if not isinstance(action, torch.Tensor):
                action = torch.from_numpy(action)
            action = action.unsqueeze(0)

        raw_obs, reward, terminated, truncated, info = self.env.step(action)

        # Update the Gaussian map with the new observation
        self._run_slam_step(raw_obs)

   ##   obs_dict = self._build_obs_dict(raw_obs, force_resample=False)

        # Flatten to scalars (legacy Gym style expected by MultiStepWrapper)
        if hasattr(terminated, 'item'):
            terminated = bool(terminated.item()) if terminated.ndim == 0 else bool(terminated[0].item())
        if hasattr(truncated, 'item'):
            truncated = bool(truncated.item()) if truncated.ndim == 0 else bool(truncated[0].item())
        if hasattr(reward, 'item'):
            reward = float(reward.item()) if reward.ndim == 0 else float(reward[0].item())

        done = bool(terminated or truncated)
        if done:
            #here ther is one global otimization step at the end of the episode , here only thr last keyframe is selected for optimization
            final_optimization_params = copy.deepcopy(self.optimization_params)
            self.gaussian_map.global_optimization(final_optimization_params,select_keyframe_num=1)

        obs_dict = self._build_obs_dict(raw_obs, force_resample=False)
        return obs_dict, float(reward), done, info

    def render(self, mode="rgb_array"):
        if self._last_rgb is None:
            return np.zeros((256, 512, 3), dtype=np.uint8)

        if self._last_gs_rgb is None:
            return self._last_rgb

        # Left: ManiSkill ground truth
        # Right: DynaGSLAM Gaussian reconstruction
        return np.concatenate(
            [self._last_rgb, self._last_gs_rgb],
            axis=1,
        )
    # ------------------------------------------------------------------
    # Optional: gsplat viewer
    # ------------------------------------------------------------------

    def _init_gsplat_viewer(self):
        """Launch a live viser viewer showing the subsampled Gaussians."""
        import viser
        from gsworld.mani_skill.utils.gsplat_viewer.gsplat_viewer import GsplatViewer
        from gsworld.mani_skill.utils.gsplat_viewer.utils_rasterize_render import (
            _viewer_render_fn, _on_connect,
        )
        from functools import partial

        device = torch.device('cuda')
        self._gs4viewer = {
            'means':      torch.zeros((self.num_gaussians, 3), device=device),
            'quats':      torch.zeros((self.num_gaussians, 4), device=device),
            'scales':     torch.zeros((self.num_gaussians, 3), device=device),
            'rgb_colors': torch.zeros((self.num_gaussians, 3), device=device),
            'opacities':  torch.zeros((self.num_gaussians,),   device=device),
        }
        server = viser.ViserServer(port=8081, verbose=False)
        self._viewer = GsplatViewer(
            server=server,
            render_fn=lambda cs, rts: _viewer_render_fn(cs, rts, self._gs4viewer, '3dgs', device),
            output_dir=None,
            mode='training',
        )
        import time; time.sleep(1)
        scene_center = [0.0, 0.0, 0.0]
        server.on_client_connect(partial(_on_connect, server=server, scene_center=scene_center))

    def _update_gsplat_viewer(self, gsplat_data: dict):
        """Atomically push a Gaussian-map snapshot into the viewer."""
        if not self.use_gsplat_viewer:
            return

        # Build a complete snapshot first. Viser renders on another thread, so
        # updating the fields one by one can mix tensors from consecutive map
        # sizes (for example, new means with old quaternions).
        with torch.no_grad():
            N = gsplat_data['gs_rotations_9d'].shape[0]
            rot_mats = gsplat_data['gs_rotations_9d'].reshape(N, 3, 3)
            viewer_snapshot = {
                'means': gsplat_data['gs_positions'].detach(),
                'quats': _matrix_to_quaternion_wxyz(rot_mats).detach(),
                'scales': gsplat_data['gs_log_scales'].detach(),
                'rgb_colors': gsplat_data['gs_rgb'].detach(),
                'opacities': torch.logit(
                    gsplat_data['gs_opacities'].view(-1).clamp(1e-6, 1 - 1e-6)
                ).detach(),
            }

        self._viewer.lock.acquire()
        try:
            self._gs4viewer = viewer_snapshot
            self._viewer.rerender(None)
        finally:
            self._viewer.lock.release()


# ---------------------------------------------------------------------------
# Quaternion / rotation utilities
# ---------------------------------------------------------------------------

def _quaternion_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert wxyz quaternions to 3×3 rotation matrices.
    q: (..., 4) tensor with layout [w, x, y, z]
    """
    w, x, y, z = q.unbind(-1)
    two_s = 2.0 / (q * q).sum(-1)
    o = torch.stack([
        1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
        1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
        1 - two_s * (x * x + y * y),
    ], dim=-1)
    return o.reshape(q.shape[:-1] + (3, 3))


def _matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3, 3) rotation matrices to (..., 4) wxyz quaternions."""
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.reshape(batch_dim + (9,)).unbind(-1)
    q_abs = torch.sqrt(torch.clamp(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1), min=0.0))
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], -1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], -1),
    ], dim=-2)
    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    indices = q_abs.argmax(dim=-1, keepdim=True)
    out = torch.gather(quat_candidates, -2, indices.unsqueeze(-1).expand(
        list(batch_dim) + [1, 4])).squeeze(-2)
    # Standardise to positive real part (w > 0)
    return torch.where(out[..., 0:1] < 0, -out, out)
