from typing import Union, Dict

import unittest
import zarr
import numpy as np
import torch
import torch.nn as nn
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.common.dict_of_tensor_mixin import DictOfTensorMixin


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] =  _fit(value, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
        else:
            self.params_dict['_default'] = _fit(data, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
    
    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)
    
    def __getitem__(self, key: str):
        params = self.params_dict[key]
        if 'per_timestep' in params:
            return PerTimestepLinearNormalizer(params)
        return SingleFieldLinearNormalizer(params)

    def __setitem__(self, key: str , value: 'SingleFieldLinearNormalizer'):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                if key not in self.params_dict:
                    continue
                params = self.params_dict[key]
                # Dispatch to per-timestep or standard normalize
                if 'per_timestep' in params:
                    result[key] = _normalize_per_timestep(value, params, forward=forward)
                else:
                    result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            if 'per_timestep' in params:
                return _normalize_per_timestep(x, params, forward=forward)
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and '_default' in self.params_dict:
            return self.params_dict['_default']['input_stats']
        
        result = dict()
        for key, value in self.params_dict.items():
            if key != '_default':
                result[key] = value['input_stats']
        return result


    def get_output_stats(self, key='_default'):
        input_stats = self.get_input_stats()
        if 'min' in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)
        
        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key:value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
            data: Union[torch.Tensor, np.ndarray, zarr.Array],
            last_n_dims=1,
            dtype=torch.float32,
            mode='limits',
            output_max=1.,
            output_min=-1.,
            range_eps=1e-4,
            fit_offset=True):
        self.params_dict = _fit(data, 
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset)
    
    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj
    
    @classmethod
    def create_manual(cls, 
            scale: Union[torch.Tensor, np.ndarray], 
            offset: Union[torch.Tensor, np.ndarray],
            input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]]):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x
        
        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype
        
        params_dict = nn.ParameterDict({
            'scale': to_tensor(scale),
            'offset': to_tensor(offset),
            'input_stats': nn.ParameterDict(
                dict_apply(input_stats_dict, to_tensor))
        })
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            'min': torch.tensor([-1], dtype=dtype),
            'max': torch.tensor([1], dtype=dtype),
            'mean': torch.tensor([0], dtype=dtype),
            'std': torch.tensor([1], dtype=dtype)
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict['input_stats']

    def get_output_stats(self):
        return dict_apply(self.params_dict['input_stats'], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)


class PerTimestepLinearNormalizer(DictOfTensorMixin):
    """
    Per-timestep, per-feature-dimension linear normalizer.
    
    Implements the TRI-LBM percentile normalization:
        yi = clamp(2 * (xi - p02) / (p98 - p02) - 1, -1.5, 1.5)
    
    Stores scale/offset of shape (T, D) (flattened to T*D for state_dict compatibility).
    Supports input shapes:
        - (T, D)       → single sample, no batch
        - (B, T, D)    → batched (e.g., agent_pos, action)
        - (T, N, D)    → single sample with N points/Gaussians
        - (B, T, N, D) → batched with N points/Gaussians
    
    Clamping to [-1.5, 1.5] is applied during normalize (forward) only.
    Unnormalize (backward) is never clamped, since the only unnormalized output 
    is actions, and the policy should be free to predict beyond training range.
    """

    @classmethod
    def create_clamped_percentile_normalizer(
        cls,
        p02: torch.Tensor,
        p98: torch.Tensor,
        n_timesteps: int,
        n_features: int,
        clamp_min: float = -1.5,
        clamp_max: float = 1.5,
        dtype: torch.dtype = torch.float32,
    ) -> 'PerTimestepLinearNormalizer':
        """
        Create a per-timestep percentile normalizer.
        
        Args:
            p02: 2nd percentile values, shape (T, D)
            p98: 98th percentile values, shape (T, D)
            n_timesteps: number of timesteps T
            n_features: number of feature dimensions D
            clamp_min: lower clamp bound (default -1.5)
            clamp_max: upper clamp bound (default 1.5)
            dtype: output dtype
        """
        assert p02.shape == (n_timesteps, n_features), \
            f"p02 shape {p02.shape} != ({n_timesteps}, {n_features})"
        assert p98.shape == p02.shape

        p02 = p02.to(dtype)
        p98 = p98.to(dtype)

        # TRI-LBM formula: y = 2 * (x - p02) / (p98 - p02) - 1
        # => y = x * scale + offset
        # where scale = 2 / (p98 - p02), offset = -2*p02/(p98-p02) - 1
        denom = torch.clamp(p98 - p02, min=1e-6)
        scale = (2.0 / denom)           # (T, D)
        offset = -scale * p02 - 1.0     # (T, D)

        params_dict = nn.ParameterDict({
            'scale': scale,                     # (T, D)
            'offset': offset,                   # (T, D)
            'per_timestep': nn.Parameter(torch.tensor([True], dtype=torch.bool), requires_grad=False),     # flag
            'n_timesteps': torch.tensor([n_timesteps], dtype=dtype),
            'n_features': torch.tensor([n_features], dtype=dtype),
            'clamp_min': torch.tensor([clamp_min], dtype=dtype),
            'clamp_max': torch.tensor([clamp_max], dtype=dtype),
            'input_stats': nn.ParameterDict({
                '2nd percentile': p02.flatten(),
                '98th percentile': p98.flatten(),
            })
        })
        for p in params_dict.parameters():
            p.requires_grad_(False)
        return cls(params_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize_per_timestep(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize_per_timestep(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict['input_stats']

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)



def _fit(data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
    assert mode in ['limits', 'gaussian']
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1,dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == 'limits':
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels 
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = - input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)
    
    # save
    this_params = nn.ParameterDict({
        'scale': scale,
        'offset': offset,
        'input_stats': nn.ParameterDict({
            'min': input_min,
            'max': input_max,
            'mean': input_mean,
            'std': input_std
        })
    })
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    assert 'scale' in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


def _normalize_per_timestep(x, params, forward=True):
    """
    Per-timestep, per-feature-dim normalization with forward clamping if clamp_min and clamp_max are given in params.
    
    Handles shapes:
        (T, D)       - single sample, no batch
        (B, T, D)    - batched
        (T, N, D)    - single sample with spatial dim (e.g., Gaussians)
        (B, T, N, D) - batched with spatial dim
    """
    assert 'scale' in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    
    scale = params['scale']             # (T, D)
    offset = params['offset']           # (T, D)
    x = x.to(device=scale.device, dtype=scale.dtype)
    
    T = int(params['n_timesteps'].item())
    D = int(params['n_features'].item())
    
    ndim = x.dim()
    if ndim == 2:
        # (T, D) - no broadcasting needed
        assert x.shape == (T, D), f"Expected ({T}, {D}), got {x.shape}"
        if forward:
            x = x * scale + offset
        else:
            x = (x - offset) / scale
    elif ndim == 3:
        if x.shape[-2] == T and x.shape[-1] == D:
            # (B, T, D) - broadcast scale (T, D) over batch
            if forward:
                x = x * scale.unsqueeze(0) + offset.unsqueeze(0)
            else:
                x = (x - offset.unsqueeze(0)) / scale.unsqueeze(0)
        else:
            # (T, N, D) - broadcast scale (T, 1, D) over spatial dim N
            assert x.shape[0] == T and x.shape[-1] == D, \
                f"Expected (T={T}, N, D={D}), got {x.shape}"
            if forward:
                x = x * scale[:, None, :] + offset[:, None, :]
            else:
                x = (x - offset[:, None, :]) / scale[:, None, :]
    elif ndim == 4:
        # (B, T, N, D) - broadcast scale (1, T, 1, D) over batch and spatial
        assert x.shape[-3] == T and x.shape[-1] == D, \
            f"Expected (B, T={T}, N, D={D}), got {x.shape}"
        if forward:
            x = x * scale[None, :, None, :] + offset[None, :, None, :]
        else:
            x = (x - offset[None, :, None, :]) / scale[None, :, None, :]
    else:
        raise ValueError(
            f"Unsupported shape {x.shape} for per-timestep normalization "
            f"with T={T}, D={D}. Expected 2D, 3D, or 4D tensor."
        )
    
    # Clamp only during normalize (forward). Unnormalize is never clamped because
    # the only unnormalized output is actions, where the policy should predict freely.
    if forward and 'clamp_min' in params and 'clamp_max' in params:
        c_min = params['clamp_min'].item()
        c_max = params['clamp_max'].item()
        x = torch.clamp(x, min=c_min, max=c_max)
    
    return x


def test():
    data = torch.zeros((100,10,9,2)).uniform_()
    data[...,0,0] = 0

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=1, fit_offset=False)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1., atol=1e-3)
    assert np.allclose(datan.min(), 0., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    data = torch.zeros((100,10,9,2)).uniform_()
    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='gaussian', last_n_dims=0)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.mean(), 0., atol=1e-3)
    assert np.allclose(datan.std(), 1., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)


    # dict
    data = torch.zeros((100,10,9,2)).uniform_()
    data[...,0,0] = 0

    normalizer = LinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    data = {
        'obs': torch.zeros((1000,128,9,2)).uniform_() * 512,
        'action': torch.zeros((1000,128,2)).uniform_() * 512
    }
    normalizer = LinearNormalizer()
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    dataun = normalizer.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)
    
    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    state_dict = normalizer.state_dict()
    n = LinearNormalizer()
    n.load_state_dict(state_dict)
    datan = n.normalize(data)
    dataun = n.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)
