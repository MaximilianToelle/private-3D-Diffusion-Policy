import torch
from pytorch3d.transforms import matrix_to_quaternion, matrix_to_euler_angles


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions [qw, qx, qy, qz] to rotation matrices.
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def pose7_to_mat(pose7):
    """[x,y,z,qw,qx,qy,qz] -> 4x4 SE(3) matrix."""
    mat = torch.zeros(pose7.shape[:-1] + (4, 4), dtype=pose7.dtype, device=pose7.device)
    mat[..., :3, :3] = quaternion_to_matrix(pose7[..., 3:7])
    mat[..., :3, 3] = pose7[..., :3]
    mat[..., 3, 3] = 1.0
    return mat

def mat_to_rot6d(R):
    """First two columns of rotation matrix as 6D repr. (..., 3, 3) -> (..., 6)"""
    return R[..., :2, :].reshape(R.shape[:-2] + (6,)).clone()

def mat_to_pose9d(mat):
    """4x4 SE(3) matrix -> [x,y,z, r1..r6]"""
    pos = mat[..., :3, 3]
    rot6d = mat_to_rot6d(mat[..., :3, :3])
    return torch.cat([pos, rot6d], dim=-1)

def rot6d_to_mat(d6):
    """6D rot -> 3x3 rot matrix."""
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = a1 / torch.linalg.norm(a1, dim=-1, keepdim=True)
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = b2 / torch.linalg.norm(b2, dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)

def pose9d_to_mat(pose9d):
    """[x,y,z, r1..r6] -> 4x4 SE(3) matrix."""
    mat = torch.zeros(pose9d.shape[:-1] + (4, 4), dtype=pose9d.dtype, device=pose9d.device)
    mat[..., :3, 3] = pose9d[..., :3]
    mat[..., :3, :3] = rot6d_to_mat(pose9d[..., 3:9])
    mat[..., 3, 3] = 1.0
    return mat

def mat_to_pose7d_quaternion(mat):
    """4x4 SE(3) matrix -> [x,y,z,qw,qx,qy,qz]"""
    pos = mat[..., :3, 3]
    # matrix_to_quaternion from pytorch3d returns (..., 4) with [qw, qx, qy, qz]
    quat_wxyz = matrix_to_quaternion(mat[..., :3, :3])
    return torch.cat([pos, quat_wxyz], dim=-1)

def mat_to_pose6d_euler(mat):
    """4x4 SE(3) matrix -> [x,y,z,euler_x,euler_y,euler_z] (ManiSkill XYZ convention)"""
    pos = mat[..., :3, 3]
    # PDEEPoseController natively expects euler angles in "XYZ" convention
    euler_xyz = matrix_to_euler_angles(mat[..., :3, :3], "XYZ")
    return torch.cat([pos, euler_xyz], dim=-1)