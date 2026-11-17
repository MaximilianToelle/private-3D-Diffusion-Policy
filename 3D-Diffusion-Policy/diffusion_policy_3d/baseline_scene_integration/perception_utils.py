"""
Perception utils shared by every baseline

Every stage of every baseline reaches for these: the offline dataset converters, the online
observation wrappers, the concrete scene mappers, and the GPU data augmentations.
"""

from typing import List, Tuple

import torch
from pytorch3d.ops import sample_farthest_points
from torch.nn.utils.rnn import pad_sequence


MAX_VALID_DEPTH_METERS = 5.0   # farther pixels are invalid, like the FFS backend's beyond its 1.2 m


def depth_to_meters(depth: torch.Tensor) -> torch.Tensor:
    """int16 millimeters, as ManiSkill, the FFS backend and the recordings all deliver, -> float32
    meters. 0 marks an invalid pixel; negative depth and anything beyond MAX_VALID_DEPTH_METERS
    (ManiSkill saturates int16 at 32767 for pixels without geometry) become invalid too."""
    depth_m = depth.to(torch.float32) / 1000.0
    depth_m[(depth_m < 0) | (depth_m > MAX_VALID_DEPTH_METERS)] = 0.0
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


GL_TO_CV_AXES = (1.0, -1.0, -1.0)  # OpenGL camera axes (x right, y up, z back) -> OpenCV (x right, y down, z forward)


def fuse_cameras_with_normals(depths_m, normals_gl, segmentations, intrinsics, cam2worlds_cv):
    """Backproject several cameras into one world-frame cloud. Per camera: depth (H, W) in meters,
    0 invalid; the renderer's unit surface normal (H, W, 3) in the OpenGL camera frame, as
    ManiSkill's normal texture delivers it; segmentation (H, W); intrinsic (3, 3) and
    cam2world_cv (4, 4). Returns points (P, 3), unit normals (P, 3) and labels (P,) of all
    valid pixels."""
    points, normals, labels = [], [], []
    for depth_m, normal_gl, segmentation, intrinsic, cam2world_cv in zip(depths_m, normals_gl, segmentations, intrinsics, cam2worlds_cv):
        camera_points, camera_labels = backproject_to_world(depth_m, intrinsic, cam2world_cv, segmentation)
        normal_cv = normal_gl[depth_m > 0] * normal_gl.new_tensor(GL_TO_CV_AXES)
        points.append(camera_points)
        normals.append(torch.nn.functional.normalize(normal_cv @ cam2world_cv[:3, :3].T.to(normal_cv.dtype), dim=-1))
        labels.append(camera_labels)
    return torch.cat(points), torch.cat(normals), torch.cat(labels)


def voxel_downsample_nearest_to_center(points, voxel_size):
    """Indices of one point per occupied voxel: the one nearest to the voxel center. The kept
    points are measured ones, not averages, so whatever is indexed with them (normals, labels)
    stays exact."""
    voxel_coordinates = points / voxel_size
    voxel_corners = torch.floor(voxel_coordinates)
    distance_to_center = ((voxel_coordinates - voxel_corners - 0.5) ** 2).sum(dim=-1)
    _, voxel_index = torch.unique(voxel_corners.long(), dim=0, return_inverse=True)
    nearest_first = torch.argsort(distance_to_center)
    grouped = nearest_first[torch.argsort(voxel_index[nearest_first], stable=True)]
    first_of_voxel = torch.ones_like(grouped, dtype=torch.bool)
    first_of_voxel[1:] = voxel_index[grouped[1:]] != voxel_index[grouped[:-1]]
    return grouped[first_of_voxel]


def fused_pcd(depths_m, normals_gl, segmentations, intrinsics, cam2worlds_cv, is_body, scene_representation, generator=None):
    """One frame of the fused_pcd_dense / fused_pcd_sparse representation
    (config/scene_representation/fused_pcd.yaml), for the converter and the observation wrapper:
    the cameras fused (fuse_cameras_with_normals), the points selected, one point per voxel and a
    uniform random subset of at most num_pool_points. Without a crop_box the selected points are
    those whose segmentation label is_body (bool lookup by label) marks. Returns points (N, 3)
    and unit normals (N, 3)."""
    points, normals, labels = fuse_cameras_with_normals(depths_m, normals_gl, segmentations, intrinsics, cam2worlds_cv)
    crop_box = scene_representation["crop_box"]
    if crop_box is None:
        selected = is_body[labels]
    else:
        box = points.new_tensor(crop_box)
        selected = ((points >= box[0]) & (points <= box[1])).all(dim=-1)
    points, normals = points[selected], normals[selected]
    kept = voxel_downsample_nearest_to_center(points, scene_representation["voxel_size"])
    kept = kept[torch.randperm(len(kept), generator=generator, device=kept.device)[:scene_representation["num_pool_points"]]]
    return points[kept], normals[kept]


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


def farthest_point_pool(points, num_pool_points):
    """The stored superset of one accumulated memory: (N, 3) -> (min(N, num_pool_points), 3) by
    deterministic farthest-point sampling, or the memory itself while it is smaller.

    A frame's memory can hold 100k points and farthest-point sampling costs time proportional
    to that count, so the converter stores this pool per frame instead of the full memory, and
    the second stage samples num_points out of it: PointCloudFPS in training, the observation
    wrapper at rollout. The wrapper runs both stages, so rollout observations follow the
    dataset's distribution."""
    if points.shape[0] <= num_pool_points:
        return points
    return farthest_point_downsample(points.unsqueeze(0), num_pool_points, random_start_point=False).squeeze(0)


def farthest_point_pools(clouds: List[torch.Tensor], num_pool_points) -> List[torch.Tensor]:
    """farthest_point_pool of many clouds in one call, for the converters: pytorch3d runs one CUDA
    block per cloud, so a batch of clouds costs little more wall time than a single one, whereas
    one call per cloud runs single blocks in sequence. Clouds larger than the pool are zero-padded
    to a common length and sampled together, with ``lengths`` keeping the padding out; smaller
    ones are returned unchanged. The result equals the per-cloud farthest_point_pool, since both
    start the greedy selection at the first point."""
    large = [index for index, cloud in enumerate(clouds) if cloud.shape[0] > num_pool_points]
    pools = list(clouds)
    if large:
        padded = pad_sequence([clouds[index] for index in large], batch_first=True)
        lengths = torch.tensor([clouds[index].shape[0] for index in large], device=padded.device)
        sampled = farthest_point_downsample(padded, num_pool_points, lengths=lengths, random_start_point=False)
        for row, index in enumerate(large):
            pools[index] = sampled[row]
    return pools
