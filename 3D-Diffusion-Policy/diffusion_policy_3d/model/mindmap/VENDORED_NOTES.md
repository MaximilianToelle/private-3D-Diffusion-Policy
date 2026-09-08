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
