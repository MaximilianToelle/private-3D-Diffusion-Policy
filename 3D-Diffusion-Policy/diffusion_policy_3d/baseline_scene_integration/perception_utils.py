"""
Perception utils shared by every baseline

Every stage of every baseline reaches for these: the offline dataset converters, the online
observation wrappers, the concrete scene mappers, and the GPU data augmentations.
"""

from typing import List, Tuple

import torch
from pytorch3d.ops import sample_farthest_points


MIN_PLAUSIBLE_DEPTH_METERS = 0.001
MAX_PLAUSIBLE_DEPTH_METERS = 20.0


def depth_to_meters(depth: torch.Tensor) -> torch.Tensor:
    """Depth image -> float32 meters, using dtype and magnitude together, because
    neither alone is sufficient: an integer dtype is always millimeters (ManiSkill
    records int16 mm; metric depth in an int would be quantized to 0/1/2), while a
    float dtype may hold either, so it falls back to magnitude (a robot workspace
    is never 50 m deep). Raises if the converted range is not a plausible depth
    image -- that means the two signals disagree and the guess would be wrong."""
    depth_m = depth.to(torch.float32)
    valid = depth_m > 0
    if not valid.any():
        return depth_m

    max_raw = float(depth_m[valid].max())
    is_millimeters = not depth.dtype.is_floating_point or max_raw > 20.0
    if is_millimeters:
        depth_m = depth_m / 1000.0

    max_m = max_raw / 1000.0 if is_millimeters else max_raw
    if not MIN_PLAUSIBLE_DEPTH_METERS <= max_m <= MAX_PLAUSIBLE_DEPTH_METERS:
        raise ValueError(
            f"depth_to_meters read {depth.dtype} as "
            f"{'millimeters' if is_millimeters else 'meters'}, giving a max depth of "
            f"{max_m:.4g} m (raw max {max_raw:.4g}), outside the plausible "
            f"[{MIN_PLAUSIBLE_DEPTH_METERS}, {MAX_PLAUSIBLE_DEPTH_METERS}] m range. "
            "The dtype and the magnitude disagree -- check the recorded depth unit."
        )
    return depth_m


def cam2world_cv_from_extrinsic_cv(extrinsic_cv: torch.Tensor) -> torch.Tensor:
    """SE3-invert a world->cam [3,4] OpenCV extrinsic into a [4,4]
    CV-convention cam2world (what depth-camera mappers expect)."""
    R = extrinsic_cv[:3, :3]
    t = extrinsic_cv[:3, 3]
    T = torch.eye(4, dtype=extrinsic_cv.dtype, device=extrinsic_cv.device)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def center_crop_square(
    images: List[torch.Tensor], intrinsics: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Center-crop HxW images (H<W) to HxH along width, shifting the principal
    point cx by the crop offset so backprojection stays correct."""
    h, w = images[0].shape[0], images[0].shape[1]
    if h == w:
        return images, intrinsics
    assert w > h, f"expected landscape images, got {h}x{w}"
    x0 = (w - h) // 2
    cropped = [img[:, x0 : x0 + h] for img in images]
    intr = intrinsics.clone()
    intr[0, 2] -= x0
    return cropped, intr


def backproject_to_world(
    depth_m: torch.Tensor,
    intrinsics: torch.Tensor,
    cam2world_cv: torch.Tensor,
    per_pixel_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backproject a depth image to a world-frame point cloud (CV pinhole),
    dropping invalid depth==0 pixels. per_pixel_labels [H,W] (e.g. segmentation
    ids) rides along, so the caller can tell which object each surviving point
    came from. Returns (points [P,3], labels [P]).

    Intrinsics are OpenCV, where an integer pixel index already IS the sample position, so
    the grid below uses it directly. A rendered image agrees: a rasterizer samples fragment i
    at window coordinate i+0.5, and a symmetric frustum absorbs that half pixel into a
    principal point of (W-1)/2. Adding a further 0.5 would count it twice and bias every
    point by half a pixel, which is 0.4 mm at 0.5 m for this wrist camera.
    """
    h, w = depth_m.shape
    vs, us = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=depth_m.device),
        torch.arange(w, dtype=torch.float32, device=depth_m.device),
        indexing="ij",
    )
    valid = depth_m > 0
    us, vs, z = us[valid], vs[valid], depth_m[valid]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    pts_cam = torch.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], dim=-1)
    pts_h = torch.cat([pts_cam, torch.ones_like(z).unsqueeze(-1)], dim=-1)
    pts_world = (cam2world_cv.to(pts_h.dtype) @ pts_h.T).T[:, :3]
    return pts_world, per_pixel_labels[valid]


def farthest_point_downsample(points, num_points, lengths=None, random_start_point=True):
    """Reduce every cloud of ``points`` (B, N, 3) to exactly (B, num_points, 3) by
    farthest-point sampling.

    Every point-based representation needs this reduction: the spatial-memory baseline
    reduces its accumulated memory, the single-step baseline the points of the current frame,
    and the PointCloudFPS augmentation a whole training batch at once.

    The signature is batched because sampling a whole batch in one call is two orders of
    magnitude cheaper than one call per cloud. ``lengths`` (B,) gives the number of real rows
    per cloud, which padded clouds need so that their padding rows never become candidates;
    a caller holding one exactly-sized cloud passes None.

    With ``random_start_point`` the greedy selection begins at a random point, so repeated
    calls on the same cloud return different subsets. Training relies on that for
    epoch-to-epoch variation, whereas validation and rollout pass False. The former to keep
    its loss comparable across epochs and the latter so that repeated evaluations of one
    checkpoint see identical observations.
    """
    smallest_cloud = points.shape[1] if lengths is None else int(lengths.min())
    assert smallest_cloud >= num_points, (
        f"cannot sample {num_points} points from a cloud of {smallest_cloud} -- pytorch3d "
        "would zero-pad silently instead of failing")
    sampled, _ = sample_farthest_points(points, lengths=lengths, K=num_points,
                                        random_start_point=random_start_point)
    return sampled
