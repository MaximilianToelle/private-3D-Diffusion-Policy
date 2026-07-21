from gym import spaces
import torch

from diffusion_policy_3d.env.maniskill.observation_wrapper.maniskill_dp3_base_obs_wrapper import ManiSkillDP3BaseObsWrapper


class WristCamGSManiskillDP3Wrapper(ManiSkillDP3BaseObsWrapper):
    """
    Wrapper that expects its underlying environment to be a `WristCamGSWorldWrapper`.
    - Eliminating all non active and low opacity Gaussians during downsampling

    Perception-specific overrides on top of `ManiSkillDP3BaseObsWrapper`: the gs_* observation
    keys, the per-step Gaussian sampling in `_get_obs_dict`, and the per-episode buffers
    cleared in `_reset_perception_state`.
    """

    def __init__(
        self,
        env,
        representation_space,
        num_gaussians=1024,
        min_opacity=0.9,
        n_action_steps=8,
        n_obs_steps=2,
    ):
        # Perception params must be set BEFORE super().__init__(), which calls
        # _perception_observation_space() (reads self.num_gaussians).
        self.num_gaussians = num_gaussians
        self.min_opacity = min_opacity
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        # self.prev_active_mask = None

        super().__init__(env, representation_space)

    def _perception_observation_space(self):
        n = self.num_gaussians
        return {
            'gs_positions': spaces.Box(
                low=-float('inf'), high=float('inf'), shape=(n, 3), dtype='float32'),
            'gs_rotations_9d': spaces.Box(
                low=-float(1), high=float(1), shape=(n, 9), dtype='float32'),
            'gs_surface_normals': spaces.Box(
                low=-float(1), high=float(1), shape=(n, 3), dtype='float32'),
            'gs_log_scales': spaces.Box(
                low=-float('inf'), high=float('inf'), shape=(n, 3), dtype='float32'),
            'gs_opacities': spaces.Box(
                low=float(0), high=float(1), shape=(n, 1), dtype='float32'),
            'gs_rgb': spaces.Box(
                low=float(0), high=float(1), shape=(n, 3), dtype='float32'),
            'gs_semantics': spaces.Box(
                low=-float('inf'), high=float('inf'), shape=(n, 1), dtype='float32'),
        }

    def _get_obs_dict(self, obs, step):
        obs_dict = {}

        obs_dict["agent_proprio"] = self._extract_agent_proprio(obs)

        gsplats = obs['sensor_data']['gsplats'][0, ...]

        # --- Track newly-active Gaussians ---
        active_gaussians_mask = gsplats[:, 19].to(torch.bool)

        # if self.prev_active_mask is None:
        #     # t=0: all currently active Gaussians are newly active
        #     newly_active_mask = active_gaussians_mask
        # else:
        #     # t>0: newly active = active now AND NOT active last step
        #     newly_active_mask = active_gaussians_mask & ~self.prev_active_mask
        # self.prev_active_mask = active_gaussians_mask  # update for next step

        # Policy outputs self.n_action_steps based on the last self.n_obs_steps
        # Example: For the init default values, the policy receives obs from timestep (0, 0), (7, 8), (15, 16), (23, 24) ...
        # We resample at 0, 7, 15, 23, ... to benefit from the gsplat consistency property over time
        # NOTE: Such a high-frequent resampling represents the extreme case
        if step == 0 or (step + self.n_obs_steps - 1) % self.n_action_steps == 0:
            assert obs['sensor_data']['gsplats'].shape[0] == 1, "Sampling needs modification to work for batched environments!"

            # TODO: torch.where does not work as soon as we introduce batched evaluation
            # Extracting all active and high opacity Gaussians in current timestep and doing sampling on top
            high_opacity_mask = (gsplats[:, 15] >= self.min_opacity)
            filter_mask = (active_gaussians_mask & high_opacity_mask)
            filtered_indices_global = torch.where(filter_mask)[0]

            # Random sampling without replacement using topk on random values
            N_filtered = filtered_indices_global.shape[0]
            assert (N_filtered >= self.num_gaussians), "Not enough Gaussians for random sampling!"
            r = torch.rand(N_filtered, device=filtered_indices_global.device)
            _, topk_indices = torch.topk(r, self.num_gaussians, largest=False)
            self.gaussian_indices = filtered_indices_global[topk_indices]

            # # FPS sampling (commented out in favor of random sampling)
            # active_pts_xyz = gsplats[active_indices_global, :3].unsqueeze(0)
            # _, sampled_indices_local = sample_farthest_points(active_pts_xyz, K=self.num_gaussians)
            # self.gaussian_indices = active_indices_global[sampled_indices_local.squeeze()]

        obs_dict["gs_positions"] = gsplats[self.gaussian_indices, :3]
        obs_dict["gs_rotations_9d"] = gsplats[self.gaussian_indices, 3:12]
        obs_dict["gs_log_scales"] = gsplats[self.gaussian_indices, 12:15]
        obs_dict["gs_opacities"] = gsplats[self.gaussian_indices, 15:16]
        obs_dict["gs_rgb"] = gsplats[self.gaussian_indices, 16:19]
        obs_dict["gs_surface_normals"] = gsplats[self.gaussian_indices, 20:23]
        obs_dict["gs_semantics"] = gsplats[self.gaussian_indices, 23:24]

        # Save RGB for rendering
        rgb = obs['sensor_data']['wrist_cam']['rgb'].squeeze(0)
        self._last_rgb = rgb.clone().to(torch.uint8)

        return obs_dict

    def _reset_perception_state(self):
        # Reset sampled gaussian indices + active tracking for a new episode.
        self.gaussian_indices = None
        # self.prev_active_mask = None
