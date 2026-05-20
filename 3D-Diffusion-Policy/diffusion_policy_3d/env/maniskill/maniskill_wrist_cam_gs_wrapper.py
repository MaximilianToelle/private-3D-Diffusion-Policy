import gym
from gym import spaces
import torch

from gsworld.mani_skill.utils.gsplat_viewer.gsplat_viewer import GsplatViewer
from gsworld.mani_skill.utils.gsplat_viewer.utils_rasterize_render import _viewer_render_fn, _on_connect

from pytorch3d.ops import sample_farthest_points


class WristCamGSManiskillDP3Wrapper(gym.Env):
    """
    Wrapper that expects its underlying environment to be a `WristCamGSWorldWrapper`.
    - Specific sampling strategy to account for newly active Gaussians over time
    """

    def __init__(
        self, 
        env, 
        num_gaussians=1024, 
        n_action_steps=8,
        n_obs_steps=2,
    ):
        super().__init__()
        self.env = env
        self.num_gaussians = num_gaussians
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        # Cast Gymnasium action space to legacy Gym action space for DP3's wrapper math
        orig_as = self.env.action_space
        self.action_space = spaces.Box(
            low=orig_as.low,
            high=orig_as.high,
            shape=orig_as.shape,
            dtype=orig_as.dtype
        )

        self.prev_active_mask = None

        obs, _ = self.env.reset()
        dummy_state = self._extract_state(obs)
        self.obs_sensor_dim = dummy_state.shape[0]

        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.obs_sensor_dim,),
                dtype='float32'
            ),
            'gs_positions': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.num_gaussians, 3), 
                dtype='float32'
            ),
            'gs_rotations_9d': spaces.Box(
                low=-float(1), high=float(1),
                shape=(self.num_gaussians, 9), 
                dtype='float32'
            ),
            'gs_surface_normals': spaces.Box(
                low=-float(1), high=float(1),
                shape=(self.num_gaussians, 3), 
                dtype='float32'
            ),
            'gs_log_scales': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.num_gaussians, 3), 
                dtype='float32'
            ),
            'gs_opacities': spaces.Box(
                low=float(0), high=float(1),
                shape=(self.num_gaussians, 1), 
                dtype='float32'
            ),
            'gs_rgb': spaces.Box(
                low=float(0), high=float(1),
                shape=(self.num_gaussians, 3), 
                dtype='float32'
            ),
        })
        
    def _extract_state(self, obs):
        # We assume the env returns unbatched shapes or batched arrays of size (1, ...)
        qpos = obs['agent']['qpos']
        if len(qpos.shape) > 1:
            qpos = qpos[0]
        return qpos.float()

    def _get_obs_dict(self, obs, step):
        obs_dict = {}
        
        state = self._extract_state(obs)
        obs_dict["agent_pos"] = state

        gsplats = obs['sensor_data']['gsplats'][0, ...]
        N_total = gsplats.shape[0]

        # --- Track newly-active Gaussians ---
        active_gaussians_mask = obs['sensor_data']['gsplats'][0, :, 19].to(torch.bool)

        if self.prev_active_mask is None:
            # t=0: all currently active Gaussians are newly active
            newly_active_mask = active_gaussians_mask
        else:
            # t>0: newly active = active now AND NOT active last step
            newly_active_mask = active_gaussians_mask & ~self.prev_active_mask
        self.prev_active_mask = active_gaussians_mask  # update for next step

        # NOTE: Policy outputs self.n_action_steps based on the last self.n_obs_steps
        # Example: For the init default values, the policy receives obs from timestep (0, 0), (7, 8), (15, 16), (23, 24) ...
        # so we need to resample at 0, 7, 15, 23, ... to benefit from the gsplat consistency property over time
        if step == 0 or (step + self.n_obs_steps - 1) % self.n_action_steps == 0:
            assert obs['sensor_data']['gsplats'].shape[0] == 1, "Sampling needs modification to work for batched environments!"
            
            # Extracting all active Gaussians in current timestep and doing farthest point sampling on top
            active_indices_global = torch.where(active_gaussians_mask)[0]

            active_pts_xyz = gsplats[active_indices_global, :3].unsqueeze(0)
            _, sampled_indices_local = sample_farthest_points(active_pts_xyz, K=self.num_gaussians)
            self.gaussian_indices = active_indices_global[sampled_indices_local.squeeze()]
        
        obs_dict["gs_positions"] = gsplats[self.gaussian_indices, :3]
        obs_dict["gs_rotations_9d"] = gsplats[self.gaussian_indices, 3:12]
        obs_dict["gs_log_scales"] = gsplats[self.gaussian_indices, 12:15]
        obs_dict["gs_opacities"] = gsplats[self.gaussian_indices, 15:16]
        obs_dict["gs_rgb"] = gsplats[self.gaussian_indices, 16:19]
        obs_dict["gs_surface_normals"] = gsplats[self.gaussian_indices, 20:23]
         
        # Save RGB for rendering
        rgb = obs['sensor_data']['wrist_cam']['rgb'].squeeze(0)
        self._last_rgb = rgb.clone().to(torch.uint8)

        return obs_dict

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
        # Reset sampled gaussian indices for new episode
        obs, info = self.env.reset(**kwargs)

        self.gaussian_indices = None
        self.prev_active_mask = None   
        
        obs_dict = self._get_obs_dict(obs, info['elapsed_steps'].item())

        return obs_dict

    def render(self, mode="rgb_array"):
        if hasattr(self, '_last_rgb'):
            return self._last_rgb.cpu().numpy()
        return torch.zeros((256, 256, 3), dtype=torch.uint8).numpy()
