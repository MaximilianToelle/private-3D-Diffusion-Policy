"""Spatial-memory point-cloud scene mapper of the spatial_memory_pcd baseline.

The mapper accumulates depth-projected points into a single point cloud that persists
over an entire episode. Because the robot links and the actors move, every point is
stored in the local frame of the rigid body it was observed on and is re-rendered at that
body's current pose, so the memory follows the scene rigidly instead of leaving motion
trails. Points that the moving wrist camera observes more than once are removed by a
voxel dedup. The voxel dedup keeps real observed points rather than voxel centroids.

The mapper deliberately does NOT downsample: ``get_scene_representation`` returns the full
memory, whose size grows over the episode, and reducing it to the fixed number of points DP3
consumes is a separate, explicit call to ``farthest_point_downsample`` of perception_utils.py.

Its two callers are the offline converter
(scripts/dataset/conversion/convert_wrist_cam_gsworld_to_spatial_memory_pcd.py) and the online
observation wrapper, so neither the accumulation nor the reduction can diverge between the
dataset a policy trains on and the observations it sees at rollout.
"""

import torch

from diffusion_policy_3d.baseline_scene_integration.base_scene_mapper import BaseSceneMapper
from diffusion_policy_3d.baseline_scene_integration.perception_utils import backproject_to_world


def _voxel_dedup_keep_points_per_rigid_body(points, rigid_body_index_per_point, voxel_size):
    """Drop every point that falls into the same voxel cell as an earlier point of the
    same rigid body. Points of different bodies are never merged, even when they share a
    cell, because the body index is folded into the voxel key. Unlike open3d's
    ``voxel_down_sample``, this function returns original coordinates instead of per-cell
    centroids, so the memory remains a faithful point representation. It is vectorized
    over all bodies at once and is therefore cheap enough to call at every step. Returns
    the surviving ``(points, rigid_body_indices)``.
    """
    voxel_coords = torch.floor(points / voxel_size).to(torch.int64)           # (N, 3)
    cell_key = torch.cat([rigid_body_index_per_point.view(-1, 1), voxel_coords], dim=1)
    _, cell_of_point = torch.unique(cell_key, dim=0, return_inverse=True)     # (N,)
    keep = _lowest_index_per_cell(cell_of_point)
    return points[keep], rigid_body_index_per_point[keep]


def _lowest_index_per_cell(cell_of_point):
    """Return the lowest point index within each occupied cell. The result is
    deterministic, and because fresh points are appended after the existing memory, a
    stored point always wins over a re-observation of the same spot.

    The scratch buffer is sized like the input rather than by the true number of cells, so
    that reading that number never forces a GPU synchronization. Cells that no point falls
    into keep the out-of-range sentinel and are filtered out.
    """
    num_input_points = cell_of_point.numel()
    device = cell_of_point.device
    lowest_index = torch.full((num_input_points,), num_input_points,
                              device=device, dtype=torch.long)
    lowest_index.scatter_reduce_(
        0, cell_of_point, torch.arange(num_input_points, device=device),
        reduce="amin", include_self=True,
    )
    return lowest_index[lowest_index != num_input_points]


def _lookup_rigid_body_indices(seg_ids, seg_id_to_rigid_body_index):
    """Translate segmentation ids into tracked-body indices, returning -1 wherever an id is
    untracked. Ids beyond the end of the lookup table are untracked, and so are negative
    ids. That second half of the test is essential, because GS writes the background as -1,
    which would otherwise index the table from the end."""
    seg_ids = seg_ids.long()
    rigid_body_index = torch.full_like(seg_ids, -1)
    in_range = (seg_ids >= 0) & (seg_ids < seg_id_to_rigid_body_index.numel())
    rigid_body_index[in_range] = seg_id_to_rigid_body_index[seg_ids[in_range]]
    return rigid_body_index


def _transform_points(points, transform_per_point):
    """Apply one rigid transform per point: ``points`` (N, 3) and ``transform_per_point``
    (N, 4, 4) give (N, 3)."""
    return (torch.einsum('nij,nj->ni', transform_per_point[:, :3, :3], points)
            + transform_per_point[:, :3, 3])


class SpatialMemoryPcdSceneMapper(BaseSceneMapper):
    """Point-cloud memory whose points rigidly follow the body they were observed on.

    The mapper tracks a fixed, ordered list of K rigid bodies, namely the robot links and
    the movable actors, and refers to each one by its index in that list. The caller owns
    that list and passes, on every frame, a dense ``seg_id_to_rigid_body_index`` lookup
    together with the (K, 4, 4) ``rigid_body_to_world_poses`` in that same index order.
    This keeps the two segmentation-id spaces, live SAPIEN ids online and GS semantic ids
    in the h5 recordings, out of the mapper entirely.

    The memory consists of two parallel arrays, which keeps re-posing and dedup fully
    vectorized and avoids a per-body Python loop in the hot path:

      _points_in_rigid_body_frame   (M, 3) each point in the frame of its own body
      _rigid_body_index_per_point   (M,)   body each point belongs to, -1 = background

    Everything untracked, such as the table and the ground, forms one further group at
    index -1. The pose stack is extended by an identity pose that this index selects, so
    those points are stored and rendered directly in the world frame, which is correct
    because they never move. When ``eliminate_background`` is set, they are discarded at
    integration time instead.
    """

    def __init__(
        self,
        voxel_size: float,
        eliminate_background: bool,
        device: str = "cuda",
    ):
        self.voxel_size = voxel_size
        self.eliminate_background = eliminate_background
        self.device = device
        self.reset()

    def reset(self) -> None:
        self._points_in_rigid_body_frame = torch.empty((0, 3), device=self.device)
        self._rigid_body_index_per_point = torch.empty((0,), device=self.device,
                                                       dtype=torch.long)

    @property
    def num_accumulated_points(self) -> int:
        return self._points_in_rigid_body_frame.shape[0]

    def integrate_frame(
        self,
        depth_m: torch.Tensor,                     # (H, W) float32 meters
        segmentation: torch.Tensor,                # (H, W) int, CALLER's id space
        intrinsics: torch.Tensor,                  # (3, 3)
        cam2world_cv: torch.Tensor,                # (4, 4) CV convention (inv extrinsic_cv)
        seg_id_to_rigid_body_index: torch.Tensor,  # (max_seg_id + 1,) long, -1 = untracked
        rigid_body_to_world_poses: torch.Tensor,   # (K, 4, 4) body poses at THIS frame
    ) -> None:
        """Fuse one preprocessed camera frame into the memory. Each caller converts its own
        depth with the shared depth_to_meters, since the dataset reads raw h5 frames while
        the wrapper reads live env observations."""
        points_world, seg_ids = backproject_to_world(
            depth_m.to(self.device),
            intrinsics.to(self.device, torch.float32),
            cam2world_cv.to(self.device, torch.float32),
            per_pixel_labels=segmentation.to(self.device),
        )
        rigid_body_index_per_point = _lookup_rigid_body_indices(
            seg_ids, seg_id_to_rigid_body_index.to(self.device))

        if self.eliminate_background:
            is_tracked = rigid_body_index_per_point >= 0
            points_world = points_world[is_tracked]
            rigid_body_index_per_point = rigid_body_index_per_point[is_tracked]

        world_to_rigid_body = torch.linalg.inv(
            self._append_background_pose(rigid_body_to_world_poses))
        new_points_in_rigid_body_frame = _transform_points(
            points_world, world_to_rigid_body[rigid_body_index_per_point])

        # Both, points in memory and newly observed points are represented in per-body local coordinates,
        # so a re-observation of a spot on a body that has moved since falls in the same voxel cell as the 
        # stored point and is deduplicated away! 
        self._points_in_rigid_body_frame, self._rigid_body_index_per_point = (
            _voxel_dedup_keep_points_per_rigid_body(
                torch.cat([self._points_in_rigid_body_frame,
                           new_points_in_rigid_body_frame]),
                torch.cat([self._rigid_body_index_per_point, rigid_body_index_per_point]),
                self.voxel_size,
            ))

    def get_scene_representation(self, rigid_body_to_world_poses: torch.Tensor) -> torch.Tensor:
        """Render the whole memory into the world frame at the current body poses, as
        (num_accumulated_points, 3) float32. That count grows over the episode, so callers
        needing a fixed size pass the result through farthest_point_downsample."""
        rigid_body_to_world_poses = self._append_background_pose(rigid_body_to_world_poses)
        return _transform_points(
            self._points_in_rigid_body_frame,
            rigid_body_to_world_poses[self._rigid_body_index_per_point])

    def _append_background_pose(self, rigid_body_to_world_poses: torch.Tensor) -> torch.Tensor:
        """Extend the pose stack from (K, 4, 4) to (K + 1, 4, 4) on the mapper's device.
        The appended identity pose is what body index -1, the background, selects."""
        return torch.cat([
            rigid_body_to_world_poses.to(self.device, torch.float32),
            torch.eye(4, device=self.device).unsqueeze(0),
        ])
