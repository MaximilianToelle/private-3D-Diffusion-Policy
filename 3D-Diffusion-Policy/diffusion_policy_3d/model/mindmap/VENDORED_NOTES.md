# Vendored mindmap (DiffuserActor) model code

Source: https://github.com/nvidia-isaac/nvblox_mindmap @ a76886df807a1afb13d260c1fc5e208257bfc768
License: NVIDIA NSCLv1 (see LICENSE.md here) — research/evaluation use, attribution
and license copy required. Original NVIDIA copyright headers are preserved in the
copied files.

Copied verbatim (import paths rewritten to this package):
- diffuser_actor.py, encoder.py, diffusion_head.py, layers.py,
  multihead_custom_attention.py, position_encodings.py   (from mindmap/diffuser_actor/)
- data_types.py            (from mindmap/data_loading/data_types.py)
- vertex_sampling.py       (from mindmap/data_loading/vertex_sampling.py)
- sample_transformer.py    (from mindmap/data_loading/sample_transformer.py; reduced
                           to GeometryAugmentor + its helpers -- the RGB/depth
                           transformers, GeometryNoiser and VertexSampler wrapper
                           are dropped, the dataset calls sample_to_n_vertices directly)
- normalization.py, relative_conversions.py, loss.py     (from mindmap/model_utils/)
- geometry/utils.py, geometry/pytorch3d_transforms.py    (from mindmap/geometry/)
- image_mask_operations.py (from mindmap/image_processing/)

Local stand-ins (so training needs neither nvblox_torch, dgl, clip nor flash_attn):
- timer.py             no-op Timer (replaces nvblox_torch.timer)
- torch_fps.py         pure-torch farthest_point_sampler (replaces dgl.geometry)
- feature_types.py     FeatureExtractorType enum + dims (replaces feature_extraction;
                       image extractors unavailable -> MESH data type only)
- distributed.py       get_rank/print_dist (replaces model_utils.distributed_training)
- tensor_visualizer.py no-op TensorVisualizer

Not vendored: multihead_flash_attention.py + converter.py (optional flash-attn
path, unused by the model), all IsaacLab/sim/datagen/training-loop code.


# Environment: why we do not use nvblox_mindmap's own stack

Upstream builds on CUDA 11.8 and layers the full IsaacSim 4.5 + IsaacLab + DGL +
catalyst environment on top, which they need for their sim datagen and their
closed-loop evaluation. We need neither. Only two stages touch nvblox at all:

  (1) offline dataset conversion, GSWorld h5 -> nvblox feature-cloud memmap
      (scripts/dataset/conversion/convert_wrist_cam_gsworld_to_nvblox_mindmap.py
      in the outer repository)
  (2) later: closed-loop eval, which will additionally need ManiSkill + GSWorld

Training the vendored DiffuserActor needs no nvblox at all -- that is what the
local stand-ins above are for -- and runs on the pre-converted dataset in the
ordinary `gsplat_policy` env.

So we drop the whole IsaacSim/IsaacLab/DGL layer and keep only nvblox_torch plus
a feature extractor, built against CUDA 12.4 to match the host toolchain rather
than upstream's 11.8. This lives in the `gsplat_policy_nvblox` conda env
(torch 2.5.1+cu124); nvblox_torch pins torch 2.4 as its minimum, so 2.5.1 on
cu124 satisfies it. An equivalent Docker image existed for a while and was
dropped once the native conda env worked, since it duplicated the same build at
25 GB.
