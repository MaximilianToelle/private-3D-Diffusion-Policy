"""DynaGSLAM scene mapper of the dynagslam baseline.

Online Gaussian-splatting SLAM: DynaGSLAM's ``Mapping`` reconstructs the Gaussian scene
incrementally from live RGB-D frames, instead of deforming a pre-scanned 3DGS model with
kinematics as the gsplat baseline does. Ground-truth simulation data replaces the three
perception modules DynaGSLAM normally requires:
    * Segmentation  -> a dynamic-object mask handed in by the caller (replaces SAM)
    * Localization  -> the camera pose handed in by the caller (use_gt_pose = True)
    * Optical flow  -> None; DynaGSLAM falls back to its pose-only association path

``get_scene_representation`` returns EVERY Gaussian of the current map in the gs_* layout the
GS policy consumes. Reducing that to the fixed ``num_gaussians`` is the caller's job, as for
every mapper in this package.

Depth convention: callers pass depth in meters (via perception_utils.depth_to_meters).
DynaGSLAM's map_preprocess() does ``depth_map = frame.original_depth * 255``, so the Camera
is built with ``depth_m / 255`` to restore meters inside the SLAM pipeline. min_depth /
max_depth in slam_args are therefore in meters.

Two callers share this class: the online observation wrapper
(env/maniskill/observation_wrapper/dynagslam/) and, once written, the offline dataset
converter -- so the reconstruction a policy trains on cannot diverge from the one it sees
at rollout. Nothing in here may depend on gym, ManiSkill or SAPIEN.
"""

import numpy as np
import torch

from diffusion_policy_3d.baseline_scene_integration.base_scene_mapper import BaseSceneMapper

# DynaGSLAM internals
from dynagslam.scene.cameras import Camera
from dynagslam.SLAM.multiprocess.mapper_dyna_eval_sam_keti import Mapping
from dynagslam.SLAM.multiprocess.tracker import Tracker
from dynagslam.SLAM.utils import move_to_gpu, move_to_cpu


SH_C0 = 0.28209479177387814  # zeroth SH band coefficient


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


class DynaGSLAMSceneMapper(BaseSceneMapper):
    """Gaussian map built online by DynaGSLAM from posed RGB-D frames.

    Lifecycle: reset() creates a fresh Mapping + Tracker for the episode, integrate_frame()
    runs one mapping step per control step, get_scene_representation() reads the map back.
    DynaGSLAM's own optimisation schedule -- local every gaussian_update_frame steps, global
    over the last global_keyframe_num keyframes on keyframes -- runs inside Mapping.mapping().
    """

    def __init__(
        self,
        slam_args,
        optimization_params,
        control_freq: float,
        device: str = "cuda",
    ):
        """
        Parameters
        ----------
        slam_args : Namespace
            Parsed DynaGSLAM config. Must have use_gt_pose=True, mode='single process'.
        optimization_params : OptimizationParams
            DynaGSLAM optimization hyper-parameters (lr, weights, …).
        control_freq : float
            Control frequency of the frame source in Hz; frame timestamps are frame_id / control_freq.
            Online this is the env's control_freq, offline the recording's.
        """
        self.slam_args = slam_args
        self.optimization_params = optimization_params
        self.control_freq = control_freq
        self.device = device

        # Force sim-compatible DynaGSLAM settings
        self.slam_args.use_gt_pose = True
        self.slam_args.mode = "single process"

        # SLAM objects — created fresh on each reset()
        self.gaussian_map = None
        self._tracker_preprocessor = None
        self._frame_id: int = 0

        # DynaGSLAM's own rendering of the map from the last integrated camera, (H, W, 3)
        # uint8 numpy. The wrapper shows it next to the ground-truth image in render().
        self.last_gs_rgb = None

    # ------------------------------------------------------------------
    # DynaGSLAM lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Instantiate a fresh Mapping + a lightweight Tracker (geometry only)."""
        self.gaussian_map = Mapping(self.slam_args)
        # Tracker is used only for map_preprocess() (normals, vertex maps).
        # No pose estimation is performed because use_gt_pose=True.
        self._tracker_preprocessor = Tracker(self.slam_args)
        self._frame_id = 0
        self.last_gs_rgb = None

    def integrate_frame(
        self,
        rgb: torch.Tensor,            # (H, W, 3) uint8
        depth_m: torch.Tensor,        # (H, W) float32 meters
        intrinsics: torch.Tensor,     # (3, 3)
        cam2world_cv: torch.Tensor,   # (4, 4) CV convention (inv extrinsic_cv)
        segmentation: torch.Tensor,   # (H, W) int, per-pixel object ids in the CALLER's id space
        dynamic_mask: np.ndarray,     # (H, W) uint8, 1 = robot link or movable actor (replaces SAM)
    ) -> None:
        """
        Execute one full DynaGSLAM mapping step for the current frame.

        Corresponds to one iteration of the inner loop in slam_eval.main():
            1. Build DynaGSLAM Camera from the frame
            2. Take the binary segmentation mask (replaces SAM)
            3. Preprocess frame geometry via Tracker.map_preprocess()
            4. Update camera pose in the frame (use_gt_pose path)
            5. Compute world-space vertex / normal maps
            6. Run Mapping.mapping() — adds / optimises Gaussians
            7. Increment _frame_id
        """
        frame = self._make_dyna_camera(rgb, depth_m, intrinsics, cam2world_cv, self._frame_id)
        seg_mask = dynamic_mask

        move_to_gpu(frame)

        # Geometry preprocessing (vertex map, normal map, confidence map)
        frame_map = self._tracker_preprocessor.map_preprocess(frame, self._frame_id)

        # With use_gt_pose=True, tracking() just copies pose_gt into frame
        # and computes world-space vertex / normal maps — no ICP / ORB needed.
        self._tracker_preprocessor.tracking(frame, frame_map, seg_mask, None)

        # added framemap to include now the segmentation for each pixel
        frame_map["object_id_map"] = segmentation.to(
            device=frame_map["vertex_map_w"].device, dtype=torch.long
        )

        # tracking() populates the world-space geometry entries.
        d = frame_map["depth_map"]
        v = frame_map["vertex_map_w"]

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
        t_old  = (self._frame_id - 1) / self.control_freq if self._frame_id > 0 else 0.0

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

            self.last_gs_rgb = gs_rgb

        self.gaussian_map.time += 1
        move_to_cpu(frame)
        self._frame_id += 1

    def _make_dyna_camera(
        self,
        rgb: torch.Tensor,
        depth_m: torch.Tensor,
        intrinsics: torch.Tensor,
        cam2world_cv: torch.Tensor,
        frame_id: int,
    ) -> Camera:
        """
        Convert one posed RGB-D frame into a DynaGSLAM Camera object.

        Camera convention (inherited from 3DGS):
            R  = w2c rotation transposed = c2w rotation   (numpy, float64)
            T  = w2c translation                            (numpy, float64)
            pose_gt = c2w 4×4 matrix                       (numpy, float64)

        Depth is stored as depth_metres / 255 so that map_preprocess()'s
        `* 255` restores metric metres.
        """
        # --- Intrinsics ---
        K = intrinsics.float()   # (3, 3) on GPU
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx_px,  cy_px  = float(K[0, 2]), float(K[1, 2])

        H = rgb.shape[0]
        W = rgb.shape[1]

        FoVx = float(2 * torch.arctan(torch.tensor(W / (2 * fx))))
        FoVy = float(2 * torch.arctan(torch.tensor(H / (2 * fy))))

        # --- Pose (ground truth from the caller) ---
        c2w_4x4 = cam2world_cv.float()
        w2c_4x4 = torch.linalg.inv(c2w_4x4)

        R = c2w_4x4[:3, :3].cpu().numpy().astype(np.float64)   # c2w rotation = w2c.T
        T = w2c_4x4[:3,  3].cpu().numpy().astype(np.float64)   # w2c translation
        pose_gt = c2w_4x4.cpu().numpy().astype(np.float64)

        # --- RGB tensor (3, H, W) float [0, 1] ---
        rgb_chw = rgb.float().permute(2, 0, 1) / 255.0   # (3, H, W)

        # --- Depth tensor (1, H, W) float, stored as metres / 255 ---
        depth_m = depth_m.float()                            # (H, W)
        depth_chw = (depth_m / 255.0).unsqueeze(0)           # (1, H, W)

        timestamp = frame_id / self.control_freq    # seconds

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

    # ------------------------------------------------------------------
    # Gaussian extraction → policy observation
    # ------------------------------------------------------------------

    def get_scene_representation(self) -> dict:
        """
        Read self.gaussian_map.global_params and reformat into policy obs keys.

        global_params keys (from Mapping):
            xyz:       (N, L, 3) positions; renderer uses level 0
            opacity:   (N, 1)    activated opacities
            scales:    (N, 3)    activated positive scales
            rotations: (N, L, 4) wxyz quaternions; renderer uses level 0
            shs:       (N, ?, 3) spherical harmonics; index [0] = DC band
            normal:    (N, 3)    unit normal = the Gaussian's smallest-scale axis

        Returns dict with policy-facing keys, over EVERY Gaussian of the map.
        """
        gp = self.gaussian_map.global_params   # dict of tensors, all on GPU

        N = gp['xyz'].shape[0]
        if N == 0:
            # Map not yet seeded — return zeros as a safe fallback
            device = torch.device(self.device)
            # TODO: requires at least num_gaussians such that policy subsampling is not crashing
            return {
                'gs_positions':    torch.zeros(1, 3, device=device),
                'gs_rotations_9d': torch.zeros(1, 9, device=device),
                'gs_log_scales':   torch.zeros(1, 3, device=device),
                'gs_opacities':    torch.zeros(1, 1, device=device),
                'gs_rgb':          torch.zeros(1, 3, device=device),
                'gs_surface_normals': torch.zeros(1, 3, device=device),
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
        # The gsplat wrapper's contract also carries per-Gaussian surface normals; DynaGSLAM
        # defines them as the smallest-scale axis of each Gaussian (GaussianPointCloud.get_normal).
        normals = gp['normal']

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
            'gs_surface_normals': normals,
        }
