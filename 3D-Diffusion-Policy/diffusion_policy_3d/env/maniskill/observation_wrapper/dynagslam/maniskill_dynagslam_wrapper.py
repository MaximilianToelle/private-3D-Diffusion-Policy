"""Live-environment half of the dynagslam baseline.

The scene mapper, ``DynaGSLAMSceneMapper`` in baseline_scene_integration/, reconstructs the
Gaussian map online from posed RGB-D frames. This wrapper is everything SAPIEN-specific
around it: pulling the frame out of the ManiSkill observation, building the dynamic-object
mask from the simulator's segmentation, and reducing the map to the fixed number of
Gaussians the policy consumes. It receives the mapper instead of constructing it (see
env_runner/maniskill_env_obs_wrapper_builder.py), so the same mapper class can later drive
the offline dataset converter.

Output observation dict (the gsplat wrapper's keys, minus gs_semantics):
    gs_positions:    (num_gaussians, 3)   float32
    gs_surface_normals: (num_gaussians, 3) float32  unit normals
    gs_rotations_9d: (num_gaussians, 9)   float32  flattened 3×3 rot matrix
    gs_log_scales:   (num_gaussians, 3)   float32
    gs_opacities:    (num_gaussians, 1)   float32  in [0, 1]
    gs_rgb:          (num_gaussians, 3)   float32  in [0, 1]
    agent_proprio:   owned by ManiSkillDP3BaseObsWrapper
"""

from typing import Optional

import numpy as np
import torch
from gym import spaces
from mani_skill.utils.structs.link import Link

from diffusion_policy_3d.baseline_scene_integration.perception_utils import (
    cam2world_cv_from_extrinsic_cv,
    depth_to_meters,
)
from diffusion_policy_3d.env.maniskill.observation_wrapper.maniskill_dp3_base_obs_wrapper import (
    ManiSkillDP3BaseObsWrapper,
)


class DynaGSLAMManiSkillDP3Wrapper(ManiSkillDP3BaseObsWrapper):
    """Observation wrapper of the dynagslam baseline.

    Perception-specific overrides on top of `ManiSkillDP3BaseObsWrapper`: the gs_* observation
    keys, one mapper step plus the per-step Gaussian subsampling in `_get_perception_obs_dict`,
    the per-episode SLAM reset in `_reset_perception_state`, and a `render` that shows the
    reconstruction next to the ground-truth image.
    """

    def __init__(
        self,
        env,
        representation_space,
        agent_proprio_dim,
        cam_name: str,
        scene_mapper,
        num_gaussians: int,
        min_opacity: float,
        use_gsplat_viewer: bool = False,
    ):
        # Perception params must be set BEFORE super().__init__() (which calls
        # _perception_observation_space, reading self.num_gaussians).
        self.cam_name = cam_name
        self.num_gaussians = num_gaussians
        self.min_opacity = min_opacity
        self.use_gsplat_viewer = use_gsplat_viewer

        super().__init__(env, representation_space, agent_proprio_dim, render_cam_name=cam_name)
        self.scene_mapper = scene_mapper

        self._gaussian_indices: Optional[torch.Tensor] = None

        if use_gsplat_viewer:
            self._init_gsplat_viewer()

    # ------------------------------------------------------------------
    # Template hooks of ManiSkillDP3BaseObsWrapper
    # ------------------------------------------------------------------

    def _perception_observation_space(self):
        return {
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
            'gs_surface_normals': spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.num_gaussians, 3), dtype='float32',
            ),
        }

    def _get_perception_obs_dict(self, obs, step):
        cam_data = obs['sensor_data'][self.cam_name]
        cam_params = obs['sensor_param'][self.cam_name]

        # ManiSkill batches sensor data over environments and a rollout runs a single one, so
        # the leading index drops that dimension and the trailing one the single depth channel.
        rgb = cam_data['rgb'][0]                                   # (H, W, 3) uint8
        depth_m = depth_to_meters(cam_data['depth'][0, ..., 0])    # (H, W) float32 meters
        segmentation = cam_data['segmentation'][0, :, :, 0]        # (H, W) int
        intrinsics = cam_params['intrinsic_cv'][0]                 # (3, 3)
        cam2world_cv = cam2world_cv_from_extrinsic_cv(cam_params['extrinsic_cv'][0])
        dynamic_mask = self._get_sim_segmentation(obs)

        self.scene_mapper.integrate_frame(
            rgb, depth_m, intrinsics, cam2world_cv, segmentation, dynamic_mask,
        )

        # Complete DynaGSLAM map.
        full_gsplat_data = self.scene_mapper.get_scene_representation()

        # Show the complete reconstruction in Viser.
        if self.use_gsplat_viewer:
            self._update_gsplat_viewer(full_gsplat_data)

        # Keep the policy input at num_gaussians, e.g. 1024.
        """
        TODO: 
            - resolve double mechanism of random sampling due to force_resample and due to Gaussian indices set to None
            - possibly leverage resampling strategy of maniskill_wrist_cam_gs_wrapper
        """
        return self._subsample_gaussians(full_gsplat_data, force_resample=(step == 0))

    def _reset_perception_state(self):
        # Fresh SLAM map for every episode. The base wrapper calls this before env.reset(); the
        # first frame is then integrated by the first _get_perception_obs_dict().
        self.scene_mapper.reset()
        self._gaussian_indices = None

    def render(self, mode="rgb_array"):
        # Left: ManiSkill ground truth
        # Right: DynaGSLAM Gaussian reconstruction
        gt_rgb = super().render(mode)
        if self.scene_mapper.last_gs_rgb is None:
            return gt_rgb
        return np.concatenate([gt_rgb, self.scene_mapper.last_gs_rgb], axis=1)

    # ------------------------------------------------------------------
    # SAPIEN -> DynaGSLAM inputs
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Map -> policy observation
    # ------------------------------------------------------------------

    def _subsample_gaussians(self, gsplat_data: dict, force_resample: bool) -> dict:
        """
        Subsample to self.num_gaussians Gaussians.

        Policy:
          • On force_resample (episode reset): filter to opacity >= min_opacity and
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
            high_mask = opacities >= self.min_opacity
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

        device = self.device
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
