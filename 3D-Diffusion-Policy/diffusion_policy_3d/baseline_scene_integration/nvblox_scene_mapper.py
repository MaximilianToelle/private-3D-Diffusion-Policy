"""Sim-agnostic nvblox spatial-memory mapper for the mindmap baseline.

Wraps nvblox_mindmap's mapping helpers behind a small interface that takes plain
tensors (depth/rgb/robot-mask/intrinsics/cam2world), so the SAME reconstruction
code drives both:
  * the offline dataset converter (scripts/dataset/conversion/convert_wrist_cam_gsworld_to_nvblox_dp3.py)
  * the online env observation wrapper (built later, via _get_perception_obs_dict/_reset_perception_state)

Internally nvblox holds a TSDF + feature voxel grid; what this class exposes is the
featurized vertex cloud the mindmap policy consumes: (vertices, vertex_features,
vertices_valid_mask). That output contract is FeatureCloudSceneMapper, defined below and
shared with the host-side mock, while the camera-geometry helpers come from
BaseSceneMapper in base_scene_mapper.py.

Heavy deps (nvblox_torch, mindmap) are imported lazily in __init__ so this module
can be imported on the host env; instantiating NvbloxSceneMapper requires the
mindmap_baseline container (docker/mindmap_baseline/).

Conventions (verified against mindmap's IsaacLabCameraHandler, which uses
quat_w_ros, i.e. the ROS optical frame):
  * cam2world must be in CV/vision convention (x right, y down, z forward).
    For ManiSkill h5 data: use inv(extrinsic_cv), NOT cam2world_gl.
  * depth in meters, float32. ManiSkill h5 stores int16 millimeters.
  * rgb uint8 HxWx3, square (center-crop 640x480 -> 480x480 and shift cx).
"""

import ctypes
import os
import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from diffusion_policy_3d.baseline_scene_integration.base_scene_mapper import BaseSceneMapper
from diffusion_policy_3d.baseline_scene_integration.perception_utils import (
    center_crop_square,
    depth_to_meters,
)


def read_dataset_perception_facts(zarr_path: str) -> dict:
    """Data-derived facts of a converted feature-cloud dataset. Reading them needs no
    baseline backend, which is why this function sits beside the mapper rather than inside
    it: workspace_bounds is a stored attribute, calibrated from the data at conversion,
    num_stored_vertices and embedding_dim are array shapes, and scene_representation is the
    record the converter stamped as a tripwire against config drift (see the env-runner
    builder). num_stored_vertices is the STORAGE superset; what the policy consumes is
    scene_representation's num_vertices."""
    import zarr

    root = zarr.open_group(zarr_path, mode="r")
    attrs = dict(root["meta"].attrs)
    assert "workspace_bounds" in attrs and "scene_representation" in attrs, (
        f"{zarr_path} lacks meta.attrs workspace_bounds/scene_representation -- "
        "reconvert with the current converter."
    )
    features_shape = root["data/vertex_features"].shape  # (T, N, C)
    return {
        "workspace_bounds": attrs["workspace_bounds"],
        "scene_representation": attrs["scene_representation"],
        "num_stored_vertices": int(features_shape[1]),
        "embedding_dim": int(features_shape[2]),
    }


class FeatureCloudSceneMapper(BaseSceneMapper):
    """
    Mappers whose scene representation is a featurized vertex cloud, mindmap-style:
    vertices [N,3] plus one feature vector [N,C] per vertex.

    Implemented by NvbloxSceneMapper below and by the host-side MockNvbloxSceneMapper in
    env/.../observation_wrapper/nvblox/. The contract lives here rather than in
    base_scene_mapper.py because it is specific to this baseline: a featurized vertex cloud
    is what the mindmap policy consumes, and no other representation produces one.

    Downsampling to the fixed vertex count the policy consumes is NOT part of this contract.
    Both the dataset and the observation wrapper call sample_to_n_vertices
    (diffusion_policy_3d/model/mindmap/vertex_sampling.py) explicitly, which is also what
    lets the dataset draw a fresh subset on every access without holding a mapper.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Channel count C of the per-vertex features get_scene_representation() returns."""

    @abstractmethod
    def integrate_frame(
        self,
        depth_m: torch.Tensor,
        rgb: torch.Tensor,
        robot_mask: torch.Tensor,
        intrinsics: torch.Tensor,
        cam2world_cv: torch.Tensor,
        decay: bool = True,
    ) -> None:
        """Fuse one camera frame into the map. depth in meters (float32), rgb uint8
        HxWx3, robot_mask bool (masked out of the static map), intrinsics [3,3],
        cam2world_cv [4,4] in CV/vision convention (x right, y down, z forward)."""

    @abstractmethod
    def get_scene_representation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the current map as (vertices [N,3] f32, vertex_features [N,C] f32), with N
        varying from frame to frame. Reducing it to the fixed count a policy consumes is a
        separate, explicit step, and there is no valid mask because every returned vertex is
        real; the mask only ever marks padding that the sampling introduces."""

    def integrate_raw_frame(
        self,
        depth: torch.Tensor,
        rgb: torch.Tensor,
        segmentation: torch.Tensor,
        intrinsics: torch.Tensor,
        cam2world_cv: torch.Tensor,
        robot_seg_ids: torch.Tensor,
        decay: bool = True,
    ) -> None:
        """Raw camera frame -> integrate_frame(), in one step. The offline converter and the
        online obs wrapper both go through HERE, so unit conversion, square cropping,
        robot masking and dtype coercion cannot diverge between the dataset a policy
        trains on and the observations it sees at rollout."""
        depth_m = depth_to_meters(depth)
        segmentation = segmentation.to(torch.int32)
        (depth_m, rgb, segmentation), intrinsics = center_crop_square(
            [depth_m, rgb, segmentation], intrinsics
        )
        robot_mask = torch.isin(segmentation, robot_seg_ids.to(segmentation.device))
        self.integrate_frame(
            depth_m, rgb.to(torch.uint8), robot_mask, intrinsics, cam2world_cv,
            decay=decay,
        )


@dataclass
class NvbloxSceneMapperConfig:
    """Duck-typed stand-in for mindmap's NvbloxMappingCfg (their helpers only read
    attributes, so we avoid their Tap/Tasks-welded constructor).

    Deliberately NO defaults: all chosen values come from
    config/scene_representation/nvblox.yaml (single source, loaded by the
    converter and composed by hydra at eval), and the aabb crop is data-derived
    """

    voxel_size_m: float
    tsdf_decay_factor: float
    projective_integrator_max_integration_distance_m: float
    projective_appearance_integrator_measurement_weight: float
    min_integration_distance_m: float

    # Post-processing crop; equals the dataset's workspace_bounds attr (which is
    # also the model's normalization bounds).
    aabb_min_m: List[float]
    aabb_max_m: List[float]

    use_dynamic_mask: bool
    upscaled_feature_image_size: Tuple[int, int]
    feature_mask_border_percent: int
    static_mask_erosion_iterations: int
    dynamic_mask_erosion_iterations: int
    valid_depth_mask_erosion_iterations: int

    def __post_init__(self):
        self.aabb_min_m = torch.as_tensor(self.aabb_min_m, dtype=torch.float32)
        self.aabb_max_m = torch.as_tensor(self.aabb_max_m, dtype=torch.float32)
        self.upscaled_feature_image_size = tuple(self.upscaled_feature_image_size)


class NvbloxSceneMapper(FeatureCloudSceneMapper):
    """Stateful spatial-memory reconstruction. integrate_frame() every control step
    (also between policy queries -- that is what accumulates memory), reset() on
    episode boundaries, get_scene_representation() whenever the policy needs an obs."""

    def __init__(
        self,
        config: NvbloxSceneMapperConfig,
        feature_type: str = "radio_v25_b",
        fpn_checkpoint: Optional[str] = None,
        device: str = "cuda",
    ):
        # libnvblox_lib.so references glog's google::InitVLOG3__ but does NOT list
        # libglog in its NEEDED entries, so the loader never pulls glog in and the
        # nvblox dlopen below dies with `undefined symbol: _ZN6google11InitVLOG3__...`.
        # Preloading glog RTLD_GLOBAL (not the default RTLD_LOCAL, whose symbols stay
        # invisible to later dlopens) puts the symbol in the process first. Located via
        # sys.prefix, NOT CONDA_PREFIX, so it also works when the env's interpreter is
        # invoked by absolute path without activation. No-op where glog is a system lib.
        glog_path = os.path.join(sys.prefix, "lib", "libglog.so.1")
        if os.path.exists(glog_path):
            ctypes.CDLL(glog_path, mode=ctypes.RTLD_GLOBAL)

        from mindmap.image_processing.feature_extraction import (
            FeatureExtractorType,
            get_feature_extractor,
        )
        from mindmap.mapping.helpers.nvblox_mapping_helpers import (
            get_nvblox_mapper,
            nvblox_integrate,
        )
        from mindmap.mapping.helpers.nvblox_output_helpers import get_vertices_and_features
        from mindmap.mapping.nvblox_mapper_constants import MAPPER_TO_ID

        self._nvblox_integrate = nvblox_integrate
        self._get_vertices_and_features = get_vertices_and_features
        self._static_mapper_id = MAPPER_TO_ID.STATIC

        self.config = config
        self.device = device

        self.mapper = get_nvblox_mapper(config)
        self.feature_extractor = get_feature_extractor(
            feature_extractor_type=FeatureExtractorType(feature_type),
            pad_to_nvblox_dim=True,
            desired_output_size=config.upscaled_feature_image_size,
            fpn_path=fpn_checkpoint,
        )

    @property
    def embedding_dim(self) -> int:
        return self.feature_extractor.embedding_dim()

    def reset(self):
        self.mapper.clear()

    def integrate_frame(
        self,
        depth_m: torch.Tensor,
        rgb: torch.Tensor,
        robot_mask: torch.Tensor,
        intrinsics: torch.Tensor,
        cam2world_cv: torch.Tensor,
        decay: bool = True,
    ):
        """
        Fuse one frame into the map. Robot pixels are masked out of the static
        map (mindmap's use_dynamic_mask); TSDF decay implements the forgetting
        that lets moved objects update.
        """
        assert depth_m.shape[0] == depth_m.shape[1], (
            f"nvblox integration requires square images, got {tuple(depth_m.shape)}; "
            "use center_crop_square first."
        )
        assert depth_m.dtype == torch.float32 and rgb.dtype == torch.uint8
        assert robot_mask.dtype == torch.bool

        if decay:
            self.mapper.decay()
        # .contiguous(): center-crop slices are strided views; nvblox rejects
        # non-contiguous tensors (live-env tensors are already on device, so no
        # implicit compaction happens in .to()).
        self._nvblox_integrate(
            mapper=self.mapper,
            nvblox_mapping_config=self.config,
            feature_extractor=self.feature_extractor,
            depth_frame=depth_m.to(self.device).contiguous(),
            intrinsics=intrinsics.to(torch.float32).cpu().contiguous(),
            camera_pose=cam2world_cv.to(torch.float32).cpu().contiguous(),
            rgb=rgb.to(self.device).contiguous(),
            dynamic_mask=robot_mask.to(self.device).contiguous(),
            include_dynamic=False,
        )

    def get_scene_representation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the whole current mesh as (vertices [N,3] f32, vertex_features [N,C] f32),
        cropped to the workspace AABB and stripped of zero-feature vertices. N therefore
        varies per frame, and the caller samples it down to the count its policy consumes."""
        # With sample_vertices=False the helper's third return is an all-ones mask, which
        # carries no information, so it is dropped here.
        vertices, features, _ = self._get_vertices_and_features(
            self.mapper,
            self._static_mapper_id,
            self.config,
            remove_zero_features=True,
            num_excess_features=self.feature_extractor.num_excess_features(),
            sample_vertices=False,
        )
        return (
            vertices.to(torch.float32).to(self.device),
            features.to(torch.float32).to(self.device),
        )
