"""Env wrapper that recolors the Gaussian scene to visualize policy attention.

Recolording changes what the env renders, which is a transformation on the env's output
Placing it in the env below `SimpleVideoRecordingWrapper` means the recolored scene is what the
video wrapper captures.

Data flow: `GaussianAttentionPlotCallback` reads the policy's per-inference max-pool
indices and pushes them here via `set_attention`; this wrapper applies the resulting
heatmap (colors + opacities) to the underlying GSWorld scene just before each step
renders. It is a no-op until fed, so it is harmless in non-attention runs.
"""

import gym
import torch

import matplotlib.pyplot as plt


class AttentionOverlayWrapper(gym.Wrapper):
    """Recolors highlighted Gaussians per policy attention before each render.

    Must wrap the GS DP3 obs wrapper directly (it reads `self.env.gaussian_indices` and
    calls the GSWorld `overwrite_gs_*` methods on `self.env.env`). Sits below the video
    recording wrapper so recolored frames are captured.

    Args:
        bg_opacity: opacity for non-highlighted Gaussians in [0, 1].
        highlight_opacity: opacity for highlighted Gaussians in [0, 1].
        colormap: matplotlib colormap name for the attention heatmap.
    """

    def __init__(self, env, bg_opacity: float = 0.1, highlight_opacity: float = 1.0,
                 colormap: str = 'cool'):
        super().__init__(env)
        self.bg_opacity = bg_opacity
        self.highlight_opacity = highlight_opacity
        self.colormap = colormap
        self._overlay = None  # (pool_idx, obs_gs_rgb) set by the attention callback

    def set_attention(self, pool_idx: torch.Tensor, obs_gs_rgb: torch.Tensor):
        """Store the latest attention indices/colors to overlay on subsequent renders."""
        self._overlay = (pool_idx, obs_gs_rgb)

    def step(self, action):
        # Apply the overlay BEFORE stepping: the GSWorld overwrite state is persistent and
        # is read by the render that happens inside `self.env.step`, so the frame the video
        # wrapper captures reflects the current attention.
        if self._overlay is not None:
            self._apply_overlay(*self._overlay)
        return self.env.step(action)

    def reset(self, **kwargs):
        self._overlay = None
        # Start each episode from the scene's original colors/opacities.
        if hasattr(self.env.env, "reset_overwritten_gs_params"):
            self.env.env.reset_overwritten_gs_params()
        return self.env.reset(**kwargs)

    def _apply_overlay(self, pool_idx: torch.Tensor, obs_gs_rgb: torch.Tensor):
        """Inject the sampled Gaussians' heatmap back into the full scene and dim the rest.

        Highlighted Gaussians become fully opaque and the background is dimmed to
        `bg_opacity`, preventing occlusion of the highlighted subset during rendering.
        """
        heatmap_rgb = compute_gs_heatmap(pool_idx, obs_gs_rgb, colormap=self.colormap)

        gs_obs_wrapper = self.env          # the GS DP3 obs wrapper
        gs_world_wrapper = gs_obs_wrapper.env

        # Map the sampled points back to indices in the full Gaussian scene
        moving_to_full_indices = torch.cat([
            gs_world_wrapper._semantic_indices[k]
            for k in gs_world_wrapper.moving_gaussians.keys()
        ])
        full_indices = moving_to_full_indices[gs_obs_wrapper.gaussian_indices]

        # Overwrite colors at the highlighted indices
        gs_world_wrapper.overwrite_gs_rgb_for_rendering(heatmap_rgb, full_indices)

        # Full-scene opacity: background dimmed, highlighted Gaussians fully opaque
        N_total = gs_world_wrapper.merged_init_gaussian_models._opacity.shape[0]
        device = full_indices.device
        all_opacities = torch.full((N_total,), self.bg_opacity, device=device)
        all_opacities[full_indices] = self.highlight_opacity
        gs_world_wrapper.overwrite_gs_opacity_for_rendering(all_opacities)


def compute_gs_heatmap(pool_indices: torch.Tensor, original_rgb: torch.Tensor, colormap='cool') -> torch.Tensor:
    """
    Args:
        pool_indices: (K) tensor from the max pool.
        original_rgb: (N, 3) tensor of the raw Gaussian colors [0, 1].
    Returns:
        heatmap_rgb: (N, 3) tensor ready for rendering.
    """
    N, _ = original_rgb.shape
    device = original_rgb.device
    heatmap_rgb = original_rgb.clone()

    cmap = plt.get_cmap(colormap)

    # 1. Count frequencies
    unique_idx, counts = torch.unique(pool_indices, return_counts=True)

    # 2. Normalize counts to [0, 1] for the colormap
    max_count = counts.max().float().clamp(min=1e-5)
    normalized_intensity = counts.float() / max_count

    # 3. Yellow tint for ignored Gaussians
    # Using standard luminance weights to get grayscale intensity
    gray = (0.299 * heatmap_rgb[:, 0] +
            0.587 * heatmap_rgb[:, 1] +
            0.114 * heatmap_rgb[:, 2])
    # Apply yellow tint (R=gray, G=gray, B=0)
    heatmap_rgb[:, 0] = gray
    heatmap_rgb[:, 1] = gray
    heatmap_rgb[:, 2] = 0.0

    # 4. Map the selected Gaussians to vibrant colors
    # cmap returns (R, G, B, A) in [0, 1]. We take RGB.
    colors_np = cmap(normalized_intensity.cpu().numpy())[:, :3]
    colors_tensor = torch.from_numpy(colors_np).to(device, dtype=torch.float32)

    # 5. Overwrite the grayscale with the heatmap colors for the winning indices
    heatmap_rgb[unique_idx, :] = colors_tensor

    return heatmap_rgb
