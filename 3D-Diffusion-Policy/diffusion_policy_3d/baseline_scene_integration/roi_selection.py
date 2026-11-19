"""
Phased region-of-interest (ROI) selection for the FMB insertion task

The engineered baseline: can a policy solve the insertion at all when it sees only the regions
the current phase of the task needs? Every point of the fused cameras belongs to at most one
region:

    fingers  fr3_leftfinger and fr3_rightfinger, by ground-truth segmentation
    piece    the piece, by ground-truth segmentation
    hole     every other point within hole_radius of the hole's axis, from below the table top to
             the board's top face: the board around the hole, its walls and the table at its bottom

The phase advances at the zero crossings of the gripper command and never goes back: GRASP until
the gripper closes, INSERT until it opens, RELEASE after. A mode names the regions each phase
samples (ACTIVE_REGIONS); every active region gets an equal fixed share of num_points, and a
region with fewer points repeats them (perception_utils.regional_farthest_point_downsample).

Regions come from the ground-truth segmentation and the ground-truth board pose, so the selection
is privileged.
"""
import numpy as np
import torch
import trimesh

from diffusion_policy_3d.baseline_scene_integration.perception_utils import (
    fuse_cameras_with_normals,
    voxel_downsample_nearest_to_center,
)
from gsplat_envs.mani_skill.utils.building.actors.fmb_object import FmbBoard, FmbObject

REGIONS = ("fingers", "piece", "hole")
FINGERS, PIECE, HOLE = range(len(REGIONS))
REGION_OF_BODY = {"fr3_leftfinger": FINGERS, "fr3_rightfinger": FINGERS, "piece": PIECE}
HOLE_MARGIN = 0.01  # m, of the hole region beyond the piece's circumradius
# world z of the hole region: from 1 cm below the table top to the board's top face (+1 mm)
HOLE_REGION_HEIGHT = (-0.01, FmbBoard.HEIGHT + 0.001)

PHASES = ("grasp", "insert", "release")
GRASP, INSERT, RELEASE = range(len(PHASES))
ACTIVE_REGIONS = {
    "targeted": [(FINGERS, PIECE, HOLE)] * len(PHASES),
    "phased": [(FINGERS, PIECE), (PIECE, HOLE), (FINGERS, PIECE)],
}


def hole_radius(piece: FmbObject) -> float:
    """Radius of the hole region: the piece's circumradius about its vertical axis plus HOLE_MARGIN."""
    vertices = trimesh.load(piece.visual_file, force="mesh").vertices
    return float(np.linalg.norm(vertices[:, :2], axis=1).max()) + HOLE_MARGIN


def region_lookup(segmentation_id_to_name: dict) -> torch.Tensor:
    """Dense lookup from segmentation id to region (REGION_OF_BODY), -1 for every other id."""
    lookup = torch.full((max(segmentation_id_to_name) + 1,), -1, dtype=torch.long)
    for segmentation_id, name in segmentation_id_to_name.items():
        lookup[segmentation_id] = REGION_OF_BODY.get(name, -1)
    return lookup


def region_indices(points, segmentation_ids, region_of_segmentation_id, hole_center_xy, hole_radius):
    """Region of every point (index into REGIONS), -1 for a point in none."""
    regions = region_of_segmentation_id[segmentation_ids]
    low, high = HOLE_REGION_HEIGHT
    in_hole_region = (((points[:, :2] - hole_center_xy).norm(dim=-1) <= hole_radius)
                      & (points[:, 2] >= low) & (points[:, 2] <= high) & (regions < 0))
    regions[in_hole_region] = HOLE
    return regions


def next_phase(phase, gripper_command):
    if phase == GRASP and gripper_command < 0:
        return INSERT
    if phase == INSERT and gripper_command > 0:
        return RELEASE
    return phase


def episode_phases(gripper_commands) -> np.ndarray:
    """Phase of each observation of an episode, observation t following the commands 0..t-1."""
    phase, phases = GRASP, []
    for command in gripper_commands:
        phases.append(phase)
        phase = next_phase(phase, command)
    return np.array(phases)


def region_budgets(mode, num_points) -> torch.Tensor:
    """(phases, regions) points per region: num_points split equally over a phase's active regions,
    the remainder to the first ones, 0 for the others."""
    budgets = torch.zeros(len(PHASES), len(REGIONS), dtype=torch.long)
    for phase, active in enumerate(ACTIVE_REGIONS[mode]):
        share, remainder = divmod(num_points, len(active))
        for rank, region in enumerate(active):
            budgets[phase, region] = share + (rank < remainder)
    return budgets


def phased_roi_pcd(depths_m, normals_gl, segmentations, intrinsics, cam2worlds_cv, region_of_segmentation_id,
                   hole_center_xy, hole_radius, scene_representation, generator=None):
    """One frame of the fused_pcd_phased representation, the pool every mode samples from, for the
    converter and the observation wrapper: the cameras fused as for fused_pcd, every point
    assigned its region, per region one point per voxel and segmented body and a uniform random
    subset of at most num_pool_points // len(REGIONS). Returns points (R, P, 3) and unit normals
    (R, P, 3), padded, and num_region_points (R,)."""
    points, normals, segmentation_ids = fuse_cameras_with_normals(depths_m, normals_gl, segmentations, intrinsics, cam2worlds_cv)
    regions = region_indices(points, segmentation_ids, region_of_segmentation_id, hole_center_xy, hole_radius)
    pool_size = scene_representation["num_pool_points"] // len(REGIONS)
    pooled_points = points.new_zeros(len(REGIONS), pool_size, 3)
    pooled_normals = points.new_zeros(len(REGIONS), pool_size, 3)
    num_region_points = torch.zeros(len(REGIONS), dtype=torch.long, device=points.device)
    for region in range(len(REGIONS)):
        in_region = torch.nonzero(regions == region).squeeze(-1)
        kept = in_region[voxel_downsample_nearest_to_center(points[in_region], scene_representation["voxel_size"],
                                                            segmentation_ids[in_region])]
        kept = kept[torch.randperm(len(kept), generator=generator, device=kept.device)[:pool_size]]
        pooled_points[region, :len(kept)] = points[kept]
        pooled_normals[region, :len(kept)] = normals[kept]
        num_region_points[region] = len(kept)
    return pooled_points, pooled_normals, num_region_points
