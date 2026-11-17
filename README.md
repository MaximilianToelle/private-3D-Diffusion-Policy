# GSplat Policy

Policy-learning half of the [gsplat_policy](../..) project: a fork of
[3D Diffusion Policy](https://3d-diffusion-policy.github.io) (DP3) rebuilt to compare **scene
representations** for manipulation. The representation is varied, everything else is held
constant: same ManiSkill task, same demonstrations, same action space, same training loop.

Upstream's docs are preserved in [DP3_README.md](DP3_README.md), [INSTALL.md](INSTALL.md) and
[ERROR_CATCH.md](ERROR_CATCH.md).

## Baselines

| representation | scene representation | experiments | scene mapper | converter |
|---|---|---|---|---|
| Gaussian splats, wrist camera | `gsworld` | `stack_gsworld` | none | `convert_wrist_cam_gsworld_to_gsplat_dp3_memmap.py` |
| Accumulated point cloud | `spatial_memory_pcd` | `stack_spatial_memory_pcd` | `SpatialMemoryPcdSceneMapper` | `convert_to_spatial_memory_pcd_memmap.py` |
| nvblox TSDF + RADIO features | `mindmap_dp3` (branch) | | `NvbloxSceneMapper` | `convert_wrist_cam_gsworld_to_nvblox_mindmap.py` |
| DynaGSLAM online Gaussians, wrist camera | `dynagslam` | `stack_dynagslam` | `DynaGSLAMSceneMapper` | none yet, see [DynaGSLAM](#dynagslam) |
| Single-frame point cloud, no memory | `single_step_pcd` | `stack_single_step_pcd` | none | `convert_to_single_step_pcd_memmap.py` |
| Point cloud fused from all cameras | `fused_pcd_dense`, `fused_pcd_sparse` | `lift_fmb_fused_pcd` | none | `convert_to_fused_pcd_memmap.py` |

Converters live in the outer repository under `scripts/dataset/conversion/`.

The nvblox row is work in progress on the `mindmap_baseline` branch, here and in the outer
repository. Master has only its lazily imported builder entry point, so it neither runs nor
breaks there.

### DynaGSLAM

- Needs `submodules/DynaGSLAM_official` on branch `dynagslam_integration`.
- Rollouts only, since there is no converter yet:
  `python -m diffusion_policy_3d.env_runner.maniskill_runner env_runner.eval_episodes=1`
- Open issues:
  - The wrapper picks Gaussians by index once per episode, but DynaGSLAM reorders them every step.
  - The observation is mostly the static table; GSWorld's holds only the robot and the objects.
  - The opacity filter has no effect: `init_opacity` 0.99 > `min_opacity` 0.98.
  - The rasterizer raises a CUDA error when no Gaussian is in view.

## Usage

Everything runs in the `gsplat_policy` conda environment, except nvblox (`gsplat_policy_nvblox`)
and DynaGSLAM (`gsplat_policy_dynagslam`).

Train from this directory, with arguments experiment, run label, seed, GPU id and any Hydra
overrides:

```bash
bash scripts/train_policy.sh lift_fmb_fused_pcd hexagon_large 42 0 task.piece=hexagon_large_h100
```

## Configs

`3D-Diffusion-Policy/diffusion_policy_3d/config/`:

| file / folder | holds |
|---|---|
| `experiment/` | one file per experiment: one task, one scene representation, one policy, its data and name |
| `task/` | the ManiSkill env: env id, piece, max_steps, shader, its callbacks |
| `scene_representation/` | what the policy sees and how: parameters stamped into the dataset, depth source, cameras, observation-wrapper builder, dataset class, point sampling, its callbacks |
| `policy/` | the network, its frame (`representation_space`), horizon, observation and action steps |
| `base.yaml` | settings every experiment shares (evaluation, training, logging), and the `dataset`, `shape_meta` and `env_runner` assembled from the selected files |
| `dp3_legacy/` | upstream's MetaWorld, Adroit, DexArt and RealDex configs, archived |

Every value is defined in exactly one file. `${merge:a,b}` in `base.yaml` joins two dictionaries
and fails if both define the same key.

Rollouts run inside training via `ManiSkillRunner`, so a training run also exercises the
environment and the wrapper chain. Dataset classes and most wrappers have a `__main__` smoke test
reporting shapes, normalizer statistics and throughput:

```bash
python -m diffusion_policy_3d.dataset.maniskill_wrist_cam_pcd_memmap_dataset
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
2. A `config/scene_representation/<name>.yaml`, whose `params` are the single source of truth for
   mapper parameters.
3. A converter in the outer repository that builds the mapper from that yaml and stamps the
   effective values into the dataset.
4. An observation wrapper in `env/maniskill/observation_wrapper/<name>/` and an entry point in
   `env_runner/maniskill_env_obs_wrapper_builder.py`.
5. Its builder, dataset class and point sampling in that same yaml, and an experiment in
   `config/experiment/`.

Conventions:

- Callers build mappers, so wrappers receive one rather than constructing it.
- The builder asserts the composed `params` equal the dataset's stamp, so editing them after
  converting fails loudly instead of evaluating on a representation the policy never saw.
- Mappers return the full representation. Downsampling to the policy's fixed size is a separate
  call: a GPU data augmentation when training, deterministic per step at rollout.
- Wrappers change env state, callbacks in `env_runner/maniskill_callbacks.py` only observe it.
- Shared helpers are module-level functions, not staticmethods, so each has one owner.

## Relation to upstream DP3

Replaces the per-representation env runners with one `ManiSkillRunner` plus injected observation
wrappers, adds the scene-mapper abstraction, and adds ManiSkill/GSWorld memmap datasets with GPU
data augmentations. Upstream's Adroit, DexArt and MetaWorld code remains untouched; their configs
are archived in `config/dp3_legacy/` and do not run with the current `train.py`.

Built on 3D Diffusion Policy by Yanjie Ze et al., MIT licensed, see [LICENSE](LICENSE).
