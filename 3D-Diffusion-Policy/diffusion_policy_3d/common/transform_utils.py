"""
Shared SE(3) pose / rotation conversion utilities.

This module is the SINGLE source of truth for converting between the pose
representations used across the pipeline:

    - pose7  : [x, y, z, qw, qx, qy, qz]              (translation + quaternion)
    - pose9d : [x, y, z, r1, r2, r3, r4, r5, r6]      (translation + 6D rotation)
    - mat    : 4x4 SE(3) homogeneous transform

Every conversion is provided in TWO parallel implementations, distinguished by a
mandatory suffix so the backend is unambiguous at every call site:

    - ``*_np``    : numpy, used inside the Dataset ``__getitem__`` /
      normalization, which runs on CPU dataloader workers.
    - ``*_torch`` : torch, used inside the env wrappers at inference time,
      which run on GPU.
"""

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, matrix_to_euler_angles


# =====================================================================
# numpy implementations (CPU dataloader workers)
# =====================================================================
def quat_wxyz_to_rotmat_np(q):
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix. Shape: (..., 4) -> (..., 3, 3).

    Self-normalizing (``two_s = 2 / (q . q)``): valid for non-unit quaternions.
    Mirrors ``quaternion_to_matrix_torch`` exactly.
    """
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / (q * q).sum(axis=-1)
    R = np.stack([
        1.0 - two_s * (j * j + k * k), two_s * (i * j - k * r),       two_s * (i * k + j * r),
        two_s * (i * j + k * r),       1.0 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r),       two_s * (j * k + i * r),       1.0 - two_s * (i * i + j * j),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    return R


def pose7_to_mat_np(pose7):
    """[x,y,z,qw,qx,qy,qz] -> 4x4 SE(3) matrix. Shape: (..., 7) -> (..., 4, 4)"""
    mat = np.zeros(pose7.shape[:-1] + (4, 4), dtype=pose7.dtype)
    mat[..., :3, :3] = quat_wxyz_to_rotmat_np(pose7[..., 3:7])
    mat[..., :3, 3] = pose7[..., :3]
    mat[..., 3, 3] = 1.0
    return mat


def mat_to_rot6d_np(R):
    """First two COLUMNS of rotation matrix as 6D repr. Shape: (..., 3, 3) -> (..., 6).

    Layout: [R[:,0], R[:,1]] = [r00, r10, r20, r01, r11, r21].
    """
    c0 = R[..., :, 0]
    c1 = R[..., :, 1]
    return np.concatenate([c0, c1], axis=-1)


def mat_to_pose9d_np(mat):
    """4x4 SE(3) matrix -> [x,y,z, r1..r6]. Shape: (..., 4, 4) -> (..., 9)"""
    pos = mat[..., :3, 3]
    rot6d = mat_to_rot6d_np(mat[..., :3, :3])
    return np.concatenate([pos, rot6d], axis=-1)


# =====================================================================
# torch implementations (GPU env wrappers)
# =====================================================================
def quaternion_to_matrix_torch(quaternions: torch.Tensor) -> torch.Tensor:
    """Quaternion [qw,qx,qy,qz] -> 3x3 rotation matrix. Shape: (..., 4) -> (..., 3, 3).

    Self-normalizing (``two_s = 2 / (q . q)``): valid for non-unit quaternions.
    Mirrors ``quat_wxyz_to_rotmat_np`` exactly.
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r),     two_s * (i * k + j * r),
            two_s * (i * j + k * r),     1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r),     two_s * (j * k + i * r),     1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def pose7_to_mat_torch(pose7):
    """[x,y,z,qw,qx,qy,qz] -> 4x4 SE(3) matrix. Shape: (..., 7) -> (..., 4, 4)"""
    mat = torch.zeros(pose7.shape[:-1] + (4, 4), dtype=pose7.dtype, device=pose7.device)
    mat[..., :3, :3] = quaternion_to_matrix_torch(pose7[..., 3:7])
    mat[..., :3, 3] = pose7[..., :3]
    mat[..., 3, 3] = 1.0
    return mat


def mat_to_rot6d_torch(R):
    """First two COLUMNS of rotation matrix as 6D repr. Shape: (..., 3, 3) -> (..., 6).

    Layout: [R[:,0], R[:,1]] = [r00, r10, r20, r01, r11, r21].
    """
    c0 = R[..., :, 0]
    c1 = R[..., :, 1]
    return torch.cat([c0, c1], dim=-1)


def mat_to_pose9d_torch(mat):
    """4x4 SE(3) matrix -> [x,y,z, r1..r6]. Shape: (..., 4, 4) -> (..., 9)"""
    pos = mat[..., :3, 3]
    rot6d = mat_to_rot6d_torch(mat[..., :3, :3])
    return torch.cat([pos, rot6d], dim=-1)


def rot6d_to_mat_torch(d6):
    """6D rot (two columns) -> 3x3 rotation matrix via Gram-Schmidt. (..., 6) -> (..., 3, 3).

    Reconstructs b1, b2, b3 as the COLUMNS of the matrix (``dim=-1``), matching
    the column-major encoding in ``mat_to_rot6d_torch`` / ``mat_to_rot6d_np``.
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = a1 / torch.linalg.norm(a1, dim=-1, keepdim=True)
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = b2 / torch.linalg.norm(b2, dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # dim=-1 => basis vectors as COLUMNS


def pose9d_to_mat_torch(pose9d):
    """[x,y,z, r1..r6] -> 4x4 SE(3) matrix. Shape: (..., 9) -> (..., 4, 4)"""
    mat = torch.zeros(pose9d.shape[:-1] + (4, 4), dtype=pose9d.dtype, device=pose9d.device)
    mat[..., :3, 3] = pose9d[..., :3]
    mat[..., :3, :3] = rot6d_to_mat_torch(pose9d[..., 3:9])
    mat[..., 3, 3] = 1.0
    return mat


def mat_to_pose7d_quaternion_torch(mat):
    """4x4 SE(3) matrix -> [x,y,z,qw,qx,qy,qz]. Shape: (..., 4, 4) -> (..., 7)"""
    pos = mat[..., :3, 3]
    # matrix_to_quaternion from pytorch3d returns (..., 4) with [qw, qx, qy, qz]
    quat_wxyz = matrix_to_quaternion(mat[..., :3, :3])
    return torch.cat([pos, quat_wxyz], dim=-1)


def mat_to_pose6d_euler_torch(mat):
    """4x4 SE(3) matrix -> [x,y,z,euler_x,euler_y,euler_z] (ManiSkill XYZ convention)."""
    pos = mat[..., :3, 3]
    # PDEEPoseController natively expects euler angles in "XYZ" convention
    euler_xyz = matrix_to_euler_angles(mat[..., :3, :3], "XYZ")
    return torch.cat([pos, euler_xyz], dim=-1)
