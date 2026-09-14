from typing import Dict
import torch
import torch.nn as nn
from diffusion_policy_3d.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer

class BasePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    def apply_torch_compile(self, mode: str = 'default'):
        """Compile the policy's compute-heavy submodules in place.

        Policies that support torch.compile override this to wrap their submodules
        while keeping the uncompiled originals registered for checkpointing (see DP3).
        The default raises: if training.use_torch_compile is set but a policy does not
        implement compilation, fail loudly rather than silently running eager.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement torch.compile "
            f"(no apply_torch_compile override), but training.use_torch_compile is enabled. "
            f"Set training.use_torch_compile=False for this policy, or implement "
            f"apply_torch_compile on it."
        )

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
