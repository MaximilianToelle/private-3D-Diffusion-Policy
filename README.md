# GSplat Policy

Policy-learning half of the [gsplat_policy](../..) project: a fork of
[3D Diffusion Policy](https://3d-diffusion-policy.github.io) (DP3) rebuilt to compare **scene
representations** for manipulation. The representation is varied, everything else is held
constant: same ManiSkill task, same demonstrations, same action space, same training loop.

Upstream's docs are preserved in [DP3_README.md](DP3_README.md), [INSTALL.md](INSTALL.md) and
[ERROR_CATCH.md](ERROR_CATCH.md).

## Baselines

| representation | train config | scene mapper | converter |
|---|---|---|---|
| Gaussian splats, wrist camera | `wrist_cam_gsplat_dp3` | none | `convert_wrist_cam_gsworld_to_gsplat_dp3_memmap.py` |
| Accumulated point cloud | `wrist_cam_spatial_memory_dp3` | `SpatialMemoryPcdSceneMapper` | `convert_wrist_cam_gsworld_to_spatial_memory_pcd_memmap.py` |
| nvblox TSDF + RADIO features | `mindmap_dp3` | `NvbloxSceneMapper` | `convert_wrist_cam_gsworld_to_nvblox_mindmap.py` |
| DynaGSLAM online Gaussians, wrist camera | `wrist_cam_dynagslam_gsplat_dp3` | `DynaGSLAMSceneMapper` | none yet -- the task's dataset keys are left `???`; rollouts only, see `DYNAGSLAM_INTEGRATION.md` |
| Single-frame point cloud | none yet | none | `convert_wrist_cam_gsworld_to_dp3.py` |

Converters live in the outer repository under `scripts/dataset/conversion/`.

The nvblox row is work in progress on the `mindmap_baseline` branch, here and in the outer
repository. Master has only its lazily imported builder entry point, so it neither runs nor
breaks there.

## Usage

Everything runs in the `gsplat_policy` conda environment, except the nvblox baseline's conversion
and evaluation, which need `nvblox_torch` and therefore `gsplat_policy_nvblox`.

Train from this directory, with arguments train config, task, run label, seed and GPU id:

```bash
bash scripts/train_policy.sh wrist_cam_gsplat_dp3 maniskill_wrist_cam_gsworld_stack full_dataset 42 0
```

Rollouts run inside training via `ManiSkillRunner`, so a training run also exercises the
environment and the wrapper chain. Dataset classes and most wrappers have a `__main__` smoke test
reporting shapes, normalizer statistics and throughput:

```bash
python -m diffusion_policy_3d.dataset.maniskill_wrist_cam_spatial_memory_pcd_memmap_dataset
```

## Adding a baseline

A baseline is one representation used twice, offline to build the dataset and online during
rollout:

```
offline   h5 recording  ->  converter  ->  memmap  ->  dataset class  ->  policy
online    live env      ->  observation wrapper  ------------------->  policy
```

Both paths drive the same **scene mapper**, a stateful reconstruction component with `reset`,
`integrate_frame` and `get_scene_representation` (`baseline_scene_integration/base_scene_mapper.py`).
One implementation serving both is what keeps training data and rollout observations identical.

A new baseline needs:

1. A mapper in `baseline_scene_integration/`, shared tensor work in its `perception_utils.py`.
2. A `config/scene_representation/<name>.yaml`, the single source of truth for mapper parameters.
3. A converter in the outer repository that builds the mapper from that yaml and stamps the
   effective values into the dataset.
4. An observation wrapper in `env/maniskill/observation_wrapper/<name>/` and an entry point in
   `env_runner/maniskill_env_obs_wrapper_builder.py`.
5. A train config and a task config in `config/`.

Conventions:

- Callers build mappers, so wrappers receive one rather than constructing it.
- The builder asserts the composed yaml equals the dataset's stamp, so editing it after
  converting fails loudly instead of evaluating on a representation the policy never saw.
- Mappers return the full representation. Downsampling to the policy's fixed size is a separate
  call: a GPU data augmentation when training, deterministic per step at rollout.
- Wrappers change env state, callbacks in `env_runner/maniskill_callbacks.py` only observe it.
- Shared helpers are module-level functions, not staticmethods, so each has one owner.

## Relation to upstream DP3

Replaces the per-representation env runners with one `ManiSkillRunner` plus injected observation
wrappers, adds the scene-mapper abstraction, and adds ManiSkill/GSWorld memmap datasets with GPU
data augmentations. Upstream's Adroit, DexArt and MetaWorld paths remain untouched.

Built on 3D Diffusion Policy by Yanjie Ze et al., MIT licensed, see [LICENSE](LICENSE).
