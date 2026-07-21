"""Base class for the ManiSkill DP3 perception wrappers.

Template-method pattern: this base owns the parts that are identical across every
perception stack (action-space cast, the step()/render() plumbing, proprioception
extraction, and the episode-reset flow). A concrete wrapper subclasses it and overrides
only what is specific to its modality:

  * `_perception_observation_space()` -> the modality's obs-space entries (gs_* / point_cloud
    / voxels ...). The base already contributes `agent_proprio`.
  * `_get_obs_dict(obs, step)` -> the produced observation dict (must include 'agent_proprio').
  * `_reset_perception_state()` -> clear any per-episode buffers (optional).

Proprioception (`agent_proprio`) is representation-dependent and shared, so it lives here.
Its declared dimension is taken from `representation_space` using the same mapping the
config's `shape_meta` uses -- NOT probed from an `env.reset()`.
"""

import gym
from gym import spaces
import torch


STATIC_ACTORS = {"table-workspace", "ground"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: read this dim from the config shape_meta instead of duplicating the mapping here.
_AGENT_PROPRIO_DIM = {"relative_ee_pose": 11, "abs_joint_pos": 9}


class ManiSkillDP3BaseObsWrapper(gym.Env):
    def __init__(self, env, representation_space):
        super().__init__()
        self.env = env
        assert representation_space in _AGENT_PROPRIO_DIM, \
            f"representation_space must be one of {tuple(_AGENT_PROPRIO_DIM)}, got '{representation_space}'"
        self.representation_space = representation_space

        # Action space: cast the underlying Gymnasium Box to a legacy Gym Box for DP3's wrapper math.
        orig_as = self.env.action_space
        self.action_space = spaces.Box(
            low=orig_as.low,
            high=orig_as.high,
            shape=orig_as.shape,
            dtype=orig_as.dtype,
        )

        # Observation space: agent_proprio (dim from representation_space) + the subclass's
        # perception keys. No env.reset() probe -- the proprio dim comes from the mapping.
        self.agent_proprio_dim = _AGENT_PROPRIO_DIM[representation_space]
        obs_space = {
            'agent_proprio': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.agent_proprio_dim,),
                dtype='float32',
            ),
        }
        obs_space.update(self._perception_observation_space())
        self.observation_space = spaces.Dict(obs_space)

    # ------------------------------------------------------------------ #
    # Template hooks -- subclasses override these.
    # ------------------------------------------------------------------ #
    def _perception_observation_space(self) -> dict:
        """Return the modality-specific obs-space entries (WITHOUT agent_proprio)."""
        raise NotImplementedError

    def _get_obs_dict(self, obs, step) -> dict:
        """Build the produced observation dict. MUST include 'agent_proprio'."""
        raise NotImplementedError

    def _reset_perception_state(self):
        """Clear any per-episode buffers. Called at the start of `reset`, before the
        first observation is produced. Default: nothing to clear."""
        pass

    # ------------------------------------------------------------------ #
    # Shared concrete behavior.
    # ------------------------------------------------------------------ #
    def _extract_agent_proprio(self, obs):
        # We assume the env returns unbatched shapes or batched arrays of size (1, ...)
        if self.representation_space == "abs_joint_pos":
            qpos = obs['agent']['qpos']
            if len(qpos.shape) > 1:
                qpos = qpos[0]
            agent_proprio = qpos.float()
        elif self.representation_space == "relative_ee_pose":
            tcp_pose = obs['extra']['tcp_pose']
            gripper_state = obs['agent']['qpos'][..., -2:]
            if len(tcp_pose.shape) > 1:
                tcp_pose = tcp_pose[0]
                gripper_state = gripper_state[0]
            agent_proprio = torch.cat([tcp_pose, gripper_state]).float()
        return agent_proprio

    def step(self, action):
        if hasattr(self.env.unwrapped, "num_envs") and self.env.unwrapped.num_envs > 0:
            if not isinstance(action, torch.Tensor):
                action = torch.from_numpy(action)
            action = action.unsqueeze(0)

        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_dict = self._get_obs_dict(obs, info['elapsed_steps'].item())

        if hasattr(terminated, "item"):
            terminated = bool(terminated.item())
        if hasattr(truncated, "item"):
            truncated = bool(truncated.item())
        if hasattr(reward, "item"):
            reward = float(reward.item())

        if isinstance(terminated, (torch.Tensor,)) and terminated.ndim > 0:
            terminated = bool(terminated[0].item())
        if isinstance(truncated, (torch.Tensor,)) and truncated.ndim > 0:
            truncated = bool(truncated[0].item())
        if isinstance(reward, (torch.Tensor,)) and reward.ndim > 0:
            reward = float(reward[0].item())

        done = bool(terminated or truncated)
        reward = float(reward)
        return obs_dict, reward, done, info

    def reset(self, **kwargs):
        # Clear per-episode state BEFORE the first observation is produced.
        self._reset_perception_state()
        obs, info = self.env.reset(**kwargs)
        return self._get_obs_dict(obs, info['elapsed_steps'].item())

    def render(self, mode="rgb_array"):
        if hasattr(self, '_last_rgb'):
            return self._last_rgb.cpu().numpy()
        return torch.zeros((256, 256, 3), dtype=torch.uint8).numpy()
