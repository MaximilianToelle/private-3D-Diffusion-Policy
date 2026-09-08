"""Stand-in for mindmap.image_processing.feature_extraction: just the extractor
type enum and feature dims, without importing clip/nvblox_torch. The actual
extractors run only during dataset conversion / closed-loop (in the
mindmap_baseline container); training consumes precomputed vertex features."""

from enum import Enum


class FeatureExtractorType(Enum):
    CLIP_RESNET50_FPN = "clip_resnet50_fpn"
    RADIO_V25_B = "radio_v25_b"
    DINO_V2_VITS14 = "dino_v2_vits14"
    RGB = "rgb"


_EMBEDDING_DIMS = {
    FeatureExtractorType.CLIP_RESNET50_FPN: 120,
    FeatureExtractorType.RADIO_V25_B: 768,
    FeatureExtractorType.DINO_V2_VITS14: 384,
    FeatureExtractorType.RGB: 3,
}


def get_nvblox_feature_dim(feature_extractor_type: FeatureExtractorType) -> int:
    return _EMBEDDING_DIMS[feature_extractor_type]


def get_feature_extractor(*args, **kwargs):
    raise NotImplementedError(
        "The vendored mindmap model supports MESH data only (precomputed vertex "
        "features). RGBD image extraction requires the real "
        "mindmap.image_processing.feature_extraction in the mindmap_baseline container."
    )
