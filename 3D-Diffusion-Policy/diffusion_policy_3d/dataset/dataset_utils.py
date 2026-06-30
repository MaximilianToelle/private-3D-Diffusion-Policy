import numpy as np


# =====================================================================
# Vectorized numpy helpers for SE(3) pose conversions
# =====================================================================
def _quat_wxyz_to_rotmat_np(q):
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix. Shape: (..., 4) -> (..., 3, 3)"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    R = np.stack([
        1.0 - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, 1.0 - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, 1.0 - (txx + tyy),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    return R


def _pose7_to_mat_np(pose7):
    """[x,y,z,qw,qx,qy,qz] -> 4x4 SE(3) matrix. Shape: (..., 7) -> (..., 4, 4)"""
    mat = np.zeros(pose7.shape[:-1] + (4, 4), dtype=pose7.dtype)
    mat[..., :3, :3] = _quat_wxyz_to_rotmat_np(pose7[..., 3:7])
    mat[..., :3, 3] = pose7[..., :3]
    mat[..., 3, 3] = 1.0
    return mat


def _mat_to_rot6d_np(R):
    """First two columns of rotation matrix as 6D repr. Shape: (..., 3, 3) -> (..., 6)"""
    return R[..., :2, :].reshape(R.shape[:-2] + (6,)).copy()


def _mat_to_pose9d_np(mat):
    """4x4 SE(3) matrix -> [x,y,z, r1..r6]. Shape: (..., 4, 4) -> (..., 9)"""
    pos = mat[..., :3, 3]
    rot6d = _mat_to_rot6d_np(mat[..., :3, :3])
    return np.concatenate([pos, rot6d], axis=-1)