"""Env wrapper that records the world pose of every scene actor after the reset and after every
step, as `SimpleVideoRecordingWrapper` records frames, so that callbacks, which run only at
action-chunk ends, can read every step."""

import gym
import numpy as np


class ActorPoseRecordingWrapper(gym.Wrapper):
    def __init__(self, env, maniskill_env):
        """maniskill_env: the unwrapped ManiSkill env whose scene holds the actors."""
        super().__init__(env)
        self.maniskill_env = maniskill_env

    def _record(self):
        self.actor_poses.append({name: actor.pose.raw_pose[0].cpu().numpy()
                                 for name, actor in self.maniskill_env.scene.actors.items()})

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.actor_poses = []
        self._record()
        return obs

    def step(self, action):
        result = super().step(action)
        self._record()
        return result

    def get_actor_poses(self, name) -> np.ndarray:
        """(steps + 1, 7) [x, y, z, qw, qx, qy, qz] of the actor, [i] before the i-th action."""
        return np.stack([poses[name] for poses in self.actor_poses])
