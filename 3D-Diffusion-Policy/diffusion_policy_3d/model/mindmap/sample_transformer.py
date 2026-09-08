# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from abc import ABC
import random
from typing import List, Tuple

import torch

from diffusion_policy_3d.model.mindmap.geometry.pytorch3d_transforms import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_multiply,
)


class SampleTransformer(ABC):
    """Base class for sample transformers"""

    def reset(self):
        """Implement this function if the transformer contains a state that should be reset before
        transforming a sample"""
        pass


class GeometryAugmentor(SampleTransformer):
    """Augment geometry input by transforming everything with the same transform, drawn from a
    uniform distribution. Call reset() to re-compute the random transform"""

    def __init__(
        self,
        random_translation_range_m: Tuple[List[float], List[float]],
        random_rpy_range_deg: Tuple[List[float], List[float]],
    ):
        """
        Args:
           random_translation_range_m: Bounds of random translation
           random_rpy_range_deg: Bounds of random rotation"""

        self._random_translation_range_m = random_translation_range_m
        self._random_rpy_range_deg = random_rpy_range_deg
        self._random_transform = None
        self.reset()

    def reset(self):
        """Recompute the transform. Should be called once before a new sample is processed"""
        if self._random_rpy_range_deg is not None and self._random_rpy_range_deg is not None:
            self._random_transform = random_transform_uniform(
                self._random_translation_range_m, self._random_rpy_range_deg
            )

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply the transform to a sample"""
        # Handle both mesh vertices and poses
        sample_tensor = sample["vertices"] if isinstance(sample, dict) else sample
        sample_tensor = apply_random_transform_to_sample(
            sample_tensor, self._random_transform[0], self._random_transform[1]
        )
        if isinstance(sample, dict):
            sample["vertices"] = sample_tensor
        else:
            sample = sample_tensor

        return sample


def random_transform_uniform(
    random_translation_range_m: Tuple[List[float], List[float]],
    random_rpy_range_deg: Tuple[List[float], List[float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random translation and quaternion.

    Args:
        random_translation_range_m (Tuple[List[float], List[float]]): The range of the random translation.
        random_rpy_range_deg (Tuple[List[float], List[float]]): The range of the random rotation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the translation vector and rotation quaternion.
    """
    # Random translation
    translation = torch.tensor(
        [
            random.uniform(random_translation_range_m[0][i], random_translation_range_m[1][i])
            for i in range(3)
        ]
    )

    # Random rotation
    rotation_degrees = torch.tensor(
        [random.uniform(random_rpy_range_deg[0][i], random_rpy_range_deg[1][i]) for i in range(3)]
    )
    rotation_radians = torch.deg2rad(rotation_degrees)
    rotation_matrix = euler_angles_to_matrix(rotation_radians, "XYZ")
    quaternion = matrix_to_quaternion(rotation_matrix)

    return translation, quaternion


def apply_random_transform_to_sample(
    sample: torch.Tensor, random_translation: torch.Tensor, random_rotation: torch.Tensor
) -> torch.Tensor:
    """
    Apply random translation and rotation to the sample.

    Args:
        sample (torch.Tensor): The sample to transform.
        random_translation (torch.Tensor): The random translation vector.
        random_rotation (torch.Tensor): The random rotation quaternion.

    Returns:
        torch.Tensor: The transformed sample.
    """
    # NOTE: initial_pose: T_AW, transformed_pose: T_BW, random_transform: T_BA
    # Either translation only or translation + quaternion + gripper state
    assert sample.shape[-1] in [3, 8]

    original_dtype = sample.dtype

    # Apply rotation to the translation part
    # B_t_BW = R_BA * A_t_AW + B_t_BA
    translation = sample[..., :3]
    transformed_translation = quaternion_apply(random_rotation, translation) + random_translation

    if sample.shape[-1] == 8:
        quaternion = sample[..., 3:7]
        gripper_state = sample[..., 7:]

        # Apply rotation to the quaternion part
        # R_BW = R_BA * R_AW
        rotated_quaternion = quaternion_multiply(random_rotation, quaternion)

        # Concatenate the transformed parts with the unchanged gripper state
        transformed_sample = torch.cat(
            [transformed_translation, rotated_quaternion, gripper_state], dim=-1
        )
    else:
        transformed_sample = transformed_translation

    assert transformed_sample.shape == sample.shape

    return transformed_sample.to(dtype=original_dtype)
