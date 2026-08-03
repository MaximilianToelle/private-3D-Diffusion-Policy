"""Facts about a live ManiSkill scene, read off the environment itself.

Every perception wrapper has to know which of the scene's rigid bodies its policy is meant
to see, and ManiSkill already answers that in ``segmentation_id_map``, which maps the integer
label a pixel carries to the ``Link`` or ``Actor`` wearing it. Reading it is a couple of
isinstance checks, but those checks encode a decision -- a robot link counts, a movable actor
counts, the table and the ground do not -- so they are answered here once for every wrapper
family rather than per wrapper.

This module is the only place where the live SAPIEN id space is interpreted. Scene mappers
never see it, since they refer to a rigid body by its index, and the offline converters answer
the same questions from the GSWorld semantic constants of the recorded h5 files instead.
"""

import torch

from mani_skill.utils.structs import Actor, Link


def tracked_rigid_bodies_by_seg_id(env):
    """The rigid bodies whose points a policy should see, keyed by their segmentation id:
    every robot link and every movable actor, with the static scenery left out.

    Callers use both halves of the mapping. The keys mask a per-pixel segmentation image,
    the values give each body's current pose. The insertion order is what fixes the index a
    scene mapper refers to a body by, so a caller that enumerates the values must do so once
    per episode and keep the result.
    """
    static_actors = env.unwrapped.STATIC_ACTOR_NAMES
    return {
        seg_id: rigid_body
        for seg_id, rigid_body in env.unwrapped.segmentation_id_map.items()
        if isinstance(rigid_body, Link)
        or (isinstance(rigid_body, Actor) and rigid_body.name not in static_actors)
    }


def robot_link_seg_ids(env):
    """Segmentation ids of the robot's links, as the int32 the nvblox robot masking expects.
    Every ``Link`` of a ManiSkill scene belongs to the robot, so no name matching is needed."""
    seg_ids = [seg_id for seg_id, rigid_body in env.unwrapped.segmentation_id_map.items()
               if isinstance(rigid_body, Link)]
    assert seg_ids, "no robot links found in segmentation_id_map"
    return torch.tensor(seg_ids, dtype=torch.int32)
