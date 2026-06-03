import torch
import torch.nn.functional as F

import open3d as o3d
import numpy as np


def compute_camera_aligned_3dgs_normals(
    rotations_9d: torch.Tensor, 
    log_scales: torch.Tensor,
    positions: torch.Tensor,
    camera_position: torch.Tensor
) -> torch.Tensor:
    """
    Computes surface normals from 3DGS parameters and possibly flips the sign of the normal to face the camera.
    
    WARNING: Only provide Gaussians which are currently visible in the camera view. 
    Otherwise, the method incorrectly flips back-face normals to point inside the object
    
    Assumptions:
        - positions and camera_position are in the SAME coordinate system 
          (e.g., the robot base frame).
        
    Args:
        rotations_9d: Tensor of shape (..., 9) representing flattened 3x3 matrices.
        log_scales: Tensor of shape (..., 3).
        positions: Tensor of shape (..., 3) representing the xyz coordinates of the Gaussians.
        camera_position: Tensor of shape (3,) representing the camera optical center.
        
    Returns:
        normals: Tensor of shape (..., 3) representing unit normals facing the camera.
    """

    original_shape = rotations_9d.shape[:-1]
    rot_matrices = rotations_9d.view(*original_shape, 3, 3)
    
    # 1. Find the shortest axis (the "flat" direction of the Gaussian)
    _, min_scale_idx = torch.min(log_scales, dim=-1)
    
    # 2. Extract the unaligned normal
    axis_selector = F.one_hot(min_scale_idx, num_classes=3).to(dtype=rot_matrices.dtype).unsqueeze(-1)
    raw_normals = torch.matmul(rot_matrices, axis_selector).squeeze(-1)
    raw_normals = F.normalize(raw_normals, p=2, dim=-1)
    
    # 3. Compute the viewing direction from the Gaussian to the camera
    view_dirs = camera_position - positions
    view_dirs = F.normalize(view_dirs, p=2, dim=-1)
    
    # 4. Compute dot product (cosine) between raw normal and viewing direction
    # We want the dot product to be positive (angle < 90 degrees)
    dot_products = torch.sum(raw_normals * view_dirs, dim=-1, keepdim=True)
    
    # 5. Flip the normals where the dot product is negative
    # If dot_product < 0, sign is -1. If > 0, sign is 1.
    # We use torch.sign, but replace 0 with 1 to prevent zeroing out perpendicular normals
    alignment_signs = torch.sign(dot_products)
    assert (alignment_signs != 0).all(), "Some normals can't be aligned as viewing angle and Gaussian normal are perpendicular!"
    # alignment_signs = torch.where(alignment_signs == 0, torch.ones_like(alignment_signs), alignment_signs)

    aligned_normals = raw_normals * alignment_signs
    
    return aligned_normals, raw_normals


def compute_3dgs_normals(rotations_9d: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    """
    Computes temporally-consistent surface normals from 3DGS parameters.
    Extracts the axis corresponding to the minimum scale (the 'flat' direction).
    
    Since rotations_9d rigidly follows the object in the environment, this normal 
    is intrinsically consistent over time without needing dynamic camera-alignment 
    that could flip signs mid-trajectory.
    
    Args:
        rotations_9d: Tensor of shape (..., 9) representing flattened 3x3 matrices
        log_scales: Tensor of shape (..., 3)
        
    Returns:
        normals: Tensor of shape (..., 3) representing unit normals.
    """
    original_shape = rotations_9d.shape[:-1]
    rot_matrices = rotations_9d.view(*original_shape, 3, 3)
    
    # 1. Find the shortest axis (the "flat" direction of the Gaussian)
    _, min_scale_idx = torch.min(log_scales, dim=-1)
    
    # Extract the specific column from the rotation matrix
    axis_selector = F.one_hot(min_scale_idx, num_classes=3).to(dtype=rot_matrices.dtype).unsqueeze(-1)
    normals = torch.matmul(rot_matrices, axis_selector).squeeze(-1)
    
    # Guarantee unit length (should already be unit length since R is orthogonal, but safe)
    normals = F.normalize(normals, p=2, dim=-1)
    
    return normals


def visualize_normals_o3d(positions: torch.Tensor, normals: torch.Tensor, rgb: torch.Tensor = None):
    """
    Blocks execution to visualize a point cloud and its surface normals using Open3D.
    Press 'Q' or close the window to resume the script!
    
    Args:
        positions: Tensor of shape (N, 3)
        normals: Tensor of shape (N, 3)
        rgb: Optional Tensor of shape (N, 3) in range [0, 1]
    """
    
    pcd = o3d.geometry.PointCloud()
    
    # Ensure inputs are 2D (N, 3)
    if len(positions.shape) > 2:
        positions = positions.view(-1, 3)
        normals = normals.view(-1, 3)
        if rgb is not None:
            rgb = rgb.view(-1, 3)
            
    pcd.points = o3d.utility.Vector3dVector(positions.detach().cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(normals.detach().cpu().numpy())
    
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy())
        
    print("Visualizing PointCloud with Normals. Close the Open3D window to continue execution...")
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)


def visualize_dual_normals_o3d(positions: torch.Tensor, normals_red: torch.Tensor, normals_green: torch.Tensor, rgb: torch.Tensor = None, length: float = 0.05):
    """
    Visualizes two sets of normals (e.g., Aligned vs Raw) with different colors in the same window.
    
    Args:
        positions: (N, 3)
        normals_red: (N, 3)
        normals_green: (N, 3)
        rgb: Optional (N, 3) for point cloud colors
        length: Length of normal lines
    """
    # Convert to numpy
    pos_np = positions.detach().cpu().numpy()
    n_red_np = normals_red.detach().cpu().numpy()
    n_green_np = normals_green.detach().cpu().numpy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos_np)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy())
    
    # Create LineSet for red normals
    lines_red = []
    points_red = []
    colors_red = []
    for i in range(len(pos_np)):
        points_red.append(pos_np[i])
        points_red.append(pos_np[i] + n_red_np[i] * length)
        lines_red.append([2*i, 2*i + 1])
        colors_red.append([1, 0, 0]) # Red
        
    ls_red = o3d.geometry.LineSet()
    ls_red.points = o3d.utility.Vector3dVector(np.array(points_red))
    ls_red.lines = o3d.utility.Vector2iVector(np.array(lines_red))
    ls_red.colors = o3d.utility.Vector3dVector(np.array(colors_red))
    
    # Create LineSet for green normals
    lines_green = []
    points_green = []
    colors_green = []
    for i in range(len(pos_np)):
        points_green.append(pos_np[i])
        points_green.append(pos_np[i] + n_green_np[i] * length)
        lines_green.append([2*i, 2*i + 1])
        colors_green.append([0, 1, 0]) # Green
        
    ls_green = o3d.geometry.LineSet()
    ls_green.points = o3d.utility.Vector3dVector(np.array(points_green))
    ls_green.lines = o3d.utility.Vector2iVector(np.array(lines_green))
    ls_green.colors = o3d.utility.Vector3dVector(np.array(colors_green))
    
    print("Visualizing dual normals (Red and Green). Close the window to continue...")
    o3d.visualization.draw_geometries([pcd, ls_red, ls_green])

