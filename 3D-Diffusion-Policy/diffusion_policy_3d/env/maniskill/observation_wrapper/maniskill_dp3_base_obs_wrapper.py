"""Base class for the ManiSkill DP3 perception wrappers.

Template-method pattern: this base owns the parts that are identical across every
perception stack (action-space cast, the step()/render() plumbing, proprioception
extraction, and the episode-reset flow). A concrete wrapper subclasses it and overrides
only what is specific to its modality:

  * `_perception_observation_space()` -> the modality's obs-space entries (gs_* / point_cloud
    / voxels ...). The base already contributes `agent_proprio`.
  * `_get_perception_obs_dict(obs, step)` -> ONLY the modality's produced keys. The base
    prepends `agent_proprio`, so subclasses never touch proprioception.
  * `_reset_perception_state()` -> clear any per-episode buffers (optional).

Proprioception (`agent_proprio`) is representation-dependent and shared, so both its obs-space
entry and its produced value are owned here -- subclasses only care about perception.
"""

import gym
from gym import spaces
import torch


class ManiSkillDP3BaseObsWrapper(gym.Env):
    def __init__(self, env, representation_space, agent_proprio_dim, render_cam_name):
        super().__init__()
        self.env = env
        self.device = self.env.unwrapped.device
        self.representation_space = representation_space
        self.agent_proprio_dim = agent_proprio_dim
        self.render_cam_name = render_cam_name
        self._last_rgb = None

        # Action space: cast the underlying Gymnasium Box to a legacy Gym Box for DP3's wrapper math.
        orig_as = self.env.action_space
        self.action_space = spaces.Box(
            low=orig_as.low,
            high=orig_as.high,
            shape=orig_as.shape,
            dtype=orig_as.dtype,
        )

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

    def _get_perception_obs_dict(self, obs, step) -> dict:
        """Return ONLY the modality's produced keys (gs_* / point_cloud / ...). The base
        prepends 'agent_proprio' in `_get_obs_dict`, so subclasses never build it."""
        raise NotImplementedError

    def _reset_perception_state(self):
        """Clear any per-episode buffers. Called at the start of `reset`, before the
        first observation is produced. Default: nothing to clear."""
        pass

    # ------------------------------------------------------------------ #
    # Shared concrete behavior.
    # ------------------------------------------------------------------ #
    def _get_obs_dict(self, obs, step) -> dict:
        """
        Assemble the full observation: 'agent_proprio' (owned here) + the subclass's perception keys.
        Also caches the last RGB frame for `render()` (shared plumbing).
        """
        self._cache_render_frame(obs)
        obs_dict = {'agent_proprio': self._extract_agent_proprio(obs)}
        obs_dict.update(self._get_perception_obs_dict(obs, step))
        return obs_dict

    def _cache_render_frame(self, obs):
        """
        Cache the last RGB frame used by `render()`. The source camera differs per
        wrapper, so each subclass passes `render_cam_name` to the base constructor.
        """
        rgb = obs['sensor_data'][self.render_cam_name]['rgb']
        if rgb.ndim == 4:   # unify batched (1, H, W, 3) across wrappers
            rgb = rgb[0]
        self._last_rgb = rgb.clone().to(torch.uint8)

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
        else:
            raise ValueError(f"Unhandled representation_space '{self.representation_space}'")
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
        if self._last_rgb is not None: 
            return self._last_rgb.cpu().numpy()
        return torch.zeros((256, 256, 3), dtype=torch.uint8).numpy()