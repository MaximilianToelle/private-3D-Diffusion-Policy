# GSplat Policy

Policy-learning half of the [gsplat_policy](../..) project: a fork of
[3D Diffusion Policy](https://3d-diffusion-policy.github.io) (DP3) rebuilt to compare **scene
representations** for manipulation.

The research question is what a policy should be shown of a scene. The representation is varied
and everything else is held constant: same ManiSkill task, same expert demonstrations, same
proprioception and action space, same training loop. Upstream's own docs are preserved in
[DP3_README.md](DP3_README.md), [INSTALL.md](INSTALL.md) and [ERROR_CATCH.md](ERROR_CATCH.md).

## Baselines

| representation | train config | scene mapper | dataset converter |
|---|---|---|---|
| Gaussian splats, wrist camera | `wrist_cam_gsplat_dp3` | none, GS wrapper only | `convert_wrist_cam_gsworld_to_gsplat_dp3_memmap.py` |
| Accumulated point cloud | `wrist_cam_spatial_memory_dp3` | `SpatialMemoryPcdSceneMapper` | `convert_wrist_cam_gsworld_to_spatial_memory_pcd_memmap.py` |
| nvblox TSDF + RADIO features | `mindmap_dp3` | `NvbloxSceneMapper` | `convert_wrist_cam_gsworld_to_nvblox_mindmap.py` |
| Single-frame point cloud | none yet | none, stateless | `convert_wrist_cam_gsworld_to_dp3.py` |

Converters live in the outer repository under `scripts/dataset/conversion/`.

The nvblox row is **work in progress on the `mindmap_baseline` branch**, together with the
matching branch of the same name in the outer repository. Master carries only its entry point in
the observation-wrapper builder, whose imports are lazy, so master neither runs nor breaks on it.
The single-frame point cloud has a task config and a builder entry point but no train config yet.

`mindmap_dp3` is standalone rather than inheriting `base_dp3`, because it drives the vendored
DiffuserActor instead of the DP3 policy and would otherwise merge a policy block it never uses.

## Usage

Two conda environments. Everything runs in `gsplat_policy`, except the nvblox baseline's
conversion and evaluation, which need `nvblox_torch` and therefore `gsplat_policy_nvblox`.

Training runs from this directory. `scripts/train_policy.sh` takes the train config, the task,
a run label, a seed and a GPU id:

```bash
conda activate gsplat_policy
bash scripts/train_policy.sh wrist_cam_spatial_memory_dp3 maniskill_wrist_cam_spatial_memory_pcd_stack full_dataset 42 0
bash scripts/train_policy.sh wrist_cam_gsplat_dp3         maniskill_wrist_cam_gs_stack                 full_dataset 42 0
```

Rollouts run inside training via `ManiSkillRunner`, so any training run exercises the
environment, the wrapper chain and the config-versus-dataset assert. `scripts/eval_policy.sh` is
upstream's inference-only path.

Each dataset class and most wrappers have a `__main__` smoke test that instantiates them against
a real dataset and reports shapes, normalizer statistics and loading throughput:

```bash
python -m diffusion_policy_3d.dataset.maniskill_wrist_cam_spatial_memory_pcd_memmap_dataset
```

## Adding a baseline

A baseline is one representation used twice, offline to build the dataset and online during
rollout. The piece that guarantees the two cannot drift apart is the **scene mapper**: a stateful
reconstruction component with `reset`, `integrate_frame` and `get_scene_representation`, defined
in `baseline_scene_integration/base_scene_mapper.py`. The same mapper class runs in both paths.

```
h5  -> scripts/dataset/conversion/convert_*.py -> memmap -> dataset/*_memmap_dataset.py -> policy
env -> env/maniskill/observation_wrapper/*/    ------------------------------------------> policy
            both go through baseline_scene_integration/<representation>_scene_mapper.py
```

What a new baseline needs:

1. A mapper in `baseline_scene_integration/`, plus any representation-independent tensor work
   added to `perception_utils.py` there.
2. A `config/scene_representation/<name>.yaml`, the single source of truth for mapper parameters.
3. A converter in the outer repository that builds the mapper from that yaml and stamps the
   effective values into the dataset.
4. An observation wrapper under `env/maniskill/observation_wrapper/<representation>/` and an
   entry point in `env_runner/maniskill_env_obs_wrapper_builder.py`.
5. A train config and a task config.

Conventions worth knowing before you extend this:

- **Mappers are built by their callers**, the converter offline and the builder online, so
  wrappers receive a mapper rather than constructing one. The builder composes the scene
  representation yaml and asserts it matches the stamp in the dataset, so editing that yaml after
  converting fails loudly instead of evaluating a policy on a representation it never saw.
- **Mappers return their full representation.** Reducing it to the fixed size the policy consumes
  is a separate call: batched on GPU as a data augmentation during training, a fresh random subset
  per epoch and a fixed one for validation, per step and deterministic during rollout.
- **Wrappers change env state, callbacks only observe it.** One `ManiSkillRunner` serves every
  baseline; logging and visualisation belong in `env_runner/maniskill_callbacks.py`.
- **Shared helpers are module-level functions, not staticmethods**, so a reader who greps for one
  finds exactly one owner.

## Layout

```
diffusion_policy_3d/
  baseline_scene_integration/  scene-mapper contract, one module per representation, perception_utils
  config/                      hydra configs: <alg>_dp3.yaml, task/, scene_representation/
  dataset/                     memmap datasets + GPU data augmentations
  env/maniskill/               observation wrappers, one directory per representation, + scene infos
  env_runner/                  ManiSkillRunner, obs-wrapper builder, eval callbacks
  model/                       DP3 policy nets (+ vendored mindmap DiffuserActor on the branch)
  policy/                      DP3 / GSplatDP3 policy wrappers (+ DiffuserActor on the branch)
train.py                       entry point for every baseline
```

## Relation to upstream DP3

This fork replaces the per-representation env runners with a single `ManiSkillRunner` plus
injected observation wrappers, adds the scene-mapper abstraction so one reconstruction
implementation serves both dataset conversion and rollout, adds ManiSkill and GSWorld datasets in
memmap format with GPU data augmentations, and adds the scene representations listed above.
Upstream's Adroit, DexArt and MetaWorld paths remain in the tree.

Built on 3D Diffusion Policy by Yanjie Ze et al., MIT licensed, see [LICENSE](LICENSE). The
upstream README with installation instructions, benchmarks and the citation is
[DP3_README.md](DP3_README.md).
