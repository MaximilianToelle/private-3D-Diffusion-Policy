"""Shared interface for baseline scene mappers.

A "scene mapper" turns per-frame camera observations into the scene representation a policy
consumes. This module holds nothing but that lifecycle contract, on which the offline dataset
converters and the online observation wrappers both depend rather than on a specific backend
(see the architecture section of the repository README). The tensor primitives every
representation builds on -- depth conversion, camera geometry, point sampling -- are functions
in perception_utils.py.

The concrete mappers each live in their own module and subclass this contract:
  * SpatialMemoryPcdSceneMapper (spatial_memory_pcd_scene_mapper.py): a point
    cloud accumulated per rigid body and rigidly re-posed.
  * NvbloxSceneMapper (nvblox_scene_mapper.py): the mindmap baseline's nvblox
    TSDF and feature voxel grid. The featurized-vertex-cloud output contract it
    shares with the host-side mock lives in that module too, since nothing
    outside that baseline implements it.
"""

from abc import ABC, abstractmethod


class BaseSceneMapper(ABC):
    """Stateful spatial-memory reconstruction contract.

    Lifecycle: reset() on episode boundaries, integrate every control step,
    query the current representation whenever the policy needs an observation.
    The integrate/query signatures live on the subclasses -- their inputs and
    outputs genuinely differ per representation.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated spatial memory (call on episode boundaries). """

    @abstractmethod
    def integrate_frame(self, *args, **kwargs) -> None:
        """
        Fuse a preprocessed camera frame into the scene representation.
        Preprocessing should be done separatly as dataset and env frames might differ.
        """

    @abstractmethod
    def get_scene_representation(self, *args, **kwargs):
        """
        Get back the current, updated scene representation without policy-specific downsampling.
        Policy-specific downsampling should be implemented as a separate method inside the specific mapper.
        """
