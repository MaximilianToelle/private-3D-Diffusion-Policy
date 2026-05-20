import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Type
from termcolor import cprint

from diffusion_policy_3d.model.vision.pointnet_extractor import create_mlp

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _shallow_mlp(in_dim: int, out_dim: int, activation: Type[nn.Module] = nn.Mish) -> nn.Sequential:
    """2-layer MLP: Linear → LayerNorm → Activation → Linear.
    
    Provides just enough non-linearity to transform a raw GS parameter space
    into a more linearly separable latent feature before merging.
    """
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),      # Normalize the linear projection
        activation(),               # Pass normalized values into Mish
        nn.Linear(out_dim, out_dim)
    )

# ──────────────────────────────────────────────────────────────────────────────
# Latent Concat Scene Encoder (Separate MLP projections before concatenation)
# ──────────────────────────────────────────────────────────────────────────────

class GSplatLatentConcatSceneEncoder(nn.Module):
    """Encodes individual Gaussian parameters with separate shallow MLPs,
    concatenates them in latent space, mixes them into a unified feature, 
    and applies a global max-pool to yield a unified scene-level feature vector. 

    GS params live in different spaces (ie R3, SO3...) which requires separate encoding before mixing.
    """

    def __init__(
        self,
        observation_space: Dict,          # Passed dynamically from the wrapper
        param_keys_feature_dims: Dict, 
        backbone_channels: List,
        out_channels: int = 64,                 # final global feature dim
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
        use_projection: bool = True,
        activation_fn: Type[nn.Module] = nn.Mish,
        **kwargs,                         # absorb extra config gracefully
    ):
        super().__init__()

        cprint(f"[GSplatLatentConcatSceneEncoder] param_groups and feature_dim={param_keys_feature_dims}", "cyan")
        cprint(f"[GSplatLatentConcatSceneEncoder] use_layernorm={use_layernorm}  final_norm={final_norm}", "cyan")
        
        # ── 1. Per-parameter group MLPs (Dynamic mapping) ───────────────────
        self.param_groups = nn.ModuleDict()
        self.ordered_keys = []
        total_group_out = 0

        # Build an MLP for each parameter group defined for encoding from the observation space:
        for key in param_keys_feature_dims.keys():
            param_group_feature_dim = param_keys_feature_dims[key]
            
            # Skip keys explicitly set to None or 0 (Hydra ablations/overrides)
            if param_group_feature_dim is None or param_group_feature_dim <= 0:
                print(f"Skipping {key} for gsplat_encoder!")
                continue

            if key in observation_space:
                self.ordered_keys.append(key)
                
                # The shape is a tuple/list (e.g., [1024, 3]), the last element is channel dim
                param_dim = observation_space[key][-1] 
                
                self.param_groups[key] = _shallow_mlp(param_dim, param_group_feature_dim, activation_fn)
                total_group_out += param_group_feature_dim
            else:
                raise ValueError(f"Expected key '{key}' not found in observation_space!")

        # ── 2. Gsplat Feature Backbone ──────────────────────────────────────
        layers = []
        current_in_dim = total_group_out
        assert backbone_channels[0] >= current_in_dim, "Mixing Layer is smaller than concatenated feature vectors"

        # Dynamically build the backbone to support any depth
        for out_dim in backbone_channels:
            layers.append(nn.Linear(current_in_dim, out_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(activation_fn())
            current_in_dim = out_dim
            
        self.gsplat_feature_backbone = nn.Sequential(*layers)

        # ── 3. Final projection (after global max-pool) ─────────────────────
        if use_projection:
            if final_norm == "layernorm":
                self.final_projection = nn.Sequential(
                    nn.Linear(backbone_channels[-1], out_channels),
                    nn.LayerNorm(out_channels),
                )
            elif final_norm == "none":
                self.final_projection = nn.Linear(backbone_channels[-1], out_channels)
            else:
                raise NotImplementedError(f"final_norm: {final_norm}")
        else:
            self.final_projection = nn.Identity()
            cprint("[GSplatLatentConcatSceneEncoder] not using final projection", "yellow")

        # Logging
        self._latest_pool_indices = None

    def forward(self, gs_data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            gs_data_dict: Dictionary mapping "gs_*" keys to (B, N, C) data tensors.
        Returns:
            (B, out_channels) global feature vector
        """
        # 1. Per-parameter groups (Ordered perfectly by initialization list)
        group_outs = []
        self._latest_feature_magnitudes = {}        # for logging
        for key in self.ordered_keys:
            out = self.param_groups[key](gs_data_dict[key])
            group_outs.append(out)

            # Compute mean L2 norm across the feature dimension (dim=-1)
            mag = out.norm(dim=-1).mean().item()
            self._latest_feature_magnitudes[f'feature_mag/{key}'] = mag

        # 2. Concatenate
        merged = torch.cat(group_outs, dim=-1)  # (B, N, total_group_out)
        
        # 3. Gsplat Feature Backbone (mixing features and expanding)
        features = self.gsplat_feature_backbone(merged)       # (B, N, backbone[-1])

        # 4. Global max-pool over Gaussians
        global_feat, self._latest_pool_indices = torch.max(features, dim=1)           # (B, backbone[-1])

        # 5. Final projection
        global_feat = self.final_projection(global_feat)      # (B, out_channels)

        return global_feat


# ──────────────────────────────────────────────────────────────────────────────
# Raw Concat Scene Encoder (PointNet Style - Direct raw concatenation)
# ──────────────────────────────────────────────────────────────────────────────

class GSplatRawConcatSceneEncoder(nn.Module):
    """Concatenates selected raw Gaussian parameters into a single vector,
    and then feeds it through a shared backbone MLP before applying global max-pool.

    PointNet style: Allows arbitrary cross-attribute correlations at the first layer.
    NOTE: Only valid for concatenating parameters which live in the same space! 
    """

    def __init__(
        self,
        observation_space: Dict,          # Passed dynamically from the wrapper
        raw_keys: List[str],              # List of parameters to concatenate raw
        backbone_channels: List,
        out_channels: int = 64,                 # final global feature dim
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
        use_projection: bool = True,
        activation_fn: Type[nn.Module] = nn.Mish,
        **kwargs,                         # absorb extra config gracefully
    ):
        super().__init__()

        self.ordered_keys = []
        in_dim = 0
        for key in raw_keys:
            if key in observation_space:
                self.ordered_keys.append(key)
                in_dim += observation_space[key][-1]
            else:
                raise ValueError(f"Expected raw key '{key}' not found in observation_space!")

        cprint(f"[GSplatRawConcatSceneEncoder] ordered_keys={self.ordered_keys} input_dim={in_dim}", "cyan")
        cprint(f"[GSplatRawConcatSceneEncoder] backbone_channels={backbone_channels} use_layernorm={use_layernorm}  final_norm={final_norm}", "cyan")

        # ── 1. Gsplat Feature Backbone ──────────────────────────────────────
        layers = []
        current_in_dim = in_dim
        assert backbone_channels[0] >= current_in_dim, "Mixing Layer is smaller than concatenated feature vectors"

        # Dynamically build the backbone to support any depth
        for out_dim in backbone_channels:
            layers.append(nn.Linear(current_in_dim, out_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(activation_fn())
            current_in_dim = out_dim
        
        # NOTE: We do LayerNorm and Activation before Max Pooling, not done in PointNetEncoder of DP3!
        self.gsplat_feature_backbone = nn.Sequential(*layers)

        # ── 2. Final projection (after global max-pool) ─────────────────────
        if use_projection:
            if final_norm == "layernorm":
                self.final_projection = nn.Sequential(
                    nn.Linear(backbone_channels[-1], out_channels),
                    nn.LayerNorm(out_channels),
                )
            elif final_norm == "none":
                self.final_projection = nn.Linear(backbone_channels[-1], out_channels)
            else:
                raise NotImplementedError(f"final_norm: {final_norm}")
        else:
            self.final_projection = nn.Identity()
            cprint("[GSplatRawConcatSceneEncoder] not using final projection", "yellow")

        # Logging
        self._latest_pool_indices = None

    def forward(self, gs_data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            gs_data_dict: Dictionary mapping "gs_*" keys to (B, N, C) data tensors.
        Returns:
            (B, out_channels) global feature vector
        """
        # 1. Concatenate raw features along channel dimension
        raw_features = [gs_data_dict[key] for key in self.ordered_keys]
        merged = torch.cat(raw_features, dim=-1)  # (B, N, in_dim)
        
        # 3. Gsplat Feature Backbone (mixing features and expanding)
        features = self.gsplat_feature_backbone(merged)       # (B, N, backbone[-1])

        # 4. Global max-pool over Gaussians
        global_feat, self._latest_pool_indices = torch.max(features, dim=1)           # (B, backbone[-1])

        # 5. Final projection
        global_feat = self.final_projection(global_feat)      # (B, out_channels)

        return global_feat


# Legacy fallback alias for backward compatibility with older checkpoint loading
GSplatSceneEncoder = GSplatLatentConcatSceneEncoder

# ──────────────────────────────────────────────────────────────────────────────
# Top-level encoder (drop-in replacement for DP3Encoder)
# ──────────────────────────────────────────────────────────────────────────────

class GSplatDP3Encoder(nn.Module):
    """
    Drop-in replacement for DP3Encoder that encodes Gaussian Splat
    parameters instead of raw point clouds, merging them with robot state.
    """

    def __init__(
        self,
        observation_space: Dict,
        out_channels: int = 64,
        use_state_features: bool = True,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn: Type[nn.Module] = nn.ReLU, 
        gsplat_encoder_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.state_key = "agent_pos"
        self.n_output_channels = out_channels

        self.state_shape = observation_space[self.state_key]

        # Extract only the GSplat parameters from the flattened observation space
        self.gs_obs_space = {
            k: v for k, v in observation_space.items() if k.startswith("gs_")
        }

        cprint(f"[GSplatDP3Encoder] gsplat param shapes: {list(self.gs_obs_space.items())}", "yellow")
        cprint(f"[GSplatDP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[GSplatDP3Encoder] use state features: {use_state_features}", "yellow")

        gsplat_encoder_cfg = dict(gsplat_encoder_cfg) if gsplat_encoder_cfg else {}
        self.use_state_features = use_state_features
        
        # Select encoder type: 'latent_concat' (default) or 'raw_concat'
        encoder_type = gsplat_encoder_cfg.pop("encoder_type", "latent_concat")
        
        if encoder_type == "latent_concat":
            # Remove raw_keys if present to avoid passing it to latent concat constructor
            gsplat_encoder_cfg.pop("raw_keys", None)
            self.extractor = GSplatLatentConcatSceneEncoder(
                observation_space=self.gs_obs_space,
                **gsplat_encoder_cfg
            )
        elif encoder_type == "raw_concat":
            # Remove param_keys_feature_dims if present to avoid passing it to raw concat constructor
            gsplat_encoder_cfg.pop("param_keys_feature_dims", None)
            self.extractor = GSplatRawConcatSceneEncoder(
                observation_space=self.gs_obs_space,
                **gsplat_encoder_cfg
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # ── State MLP (same as DP3Encoder) ───────────────────────────────────
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(
            *create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn)
        )

        cprint(f"[GSplatDP3Encoder] final output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # ── Gaussian splat features ──────────────────────────────────────────
        # Extract the Gaussian dictionaries specific to the inner extractor
        gs_data_dict = {key: observations[key] for key in self.extractor.ordered_keys}
        
        # Verify shape (B, N, C)
        for k, v in gs_data_dict.items():
            assert len(v.shape) == 3, f"Key '{k}' has shape {v.shape}, expected (B, N, C)"

        # Sort log scales to: [log_s_max, log_s_min, log_s_surface_normal]
        # NOTE: Only works because normalization stats are computed over all log scales, ie the distinct ordering of value magnitudes remains
        if 'gs_log_scales' in gs_data_dict:
            gs_data_dict['gs_log_scales'], _ = torch.sort(gs_data_dict['gs_log_scales'], dim=-1, descending=True, stable=True)

        gs_feat = self.extractor(gs_data_dict)  # (B, out_channel)

        # ── State features ───────────────────────────────────────────────────
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)   # (B, state_dim)
        
        if not self.use_state_features:
            state_feat = torch.zeros_like(state_feat)

        # Combine visual and proprioceptive context
        final_feat = torch.cat([gs_feat, state_feat], dim=-1)
        return final_feat

    def output_shape(self) -> int:
        return self.n_output_channels
