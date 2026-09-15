# DynaGSLAM integration — port onto the refactored master

`dynagslam_integration` was written against a DP3 layout with one env_runner and one flat wrapper per
scene representation. Master replaced that (commit `18516fc`, "BIG REFACTOR") with a single
`ManiSkillRunner`, perception-specific observation wrappers on a common base
(`ManiSkillDP3BaseObsWrapper`), and scene mappers in `baseline_scene_integration/` that hold the
reconstruction itself. This note records, change by change, how the DynaGSLAM code was moved into that
structure, so every deviation from the original can be checked.

**What is compared.** Fawad's last own commit `fdeaf60` (2026-09-11) against the working tree after the
merge commit `b4b0848` (origin/master merged into `dynagslam_integration`, 2026-09-14). His branch changed
seven files relative to master: `env/maniskill/dynagslam_wrapper.py` (701 lines),
`env_runner/dynagslam_maniskill_runner.py` (251), `config/task/maniskill_wrist_cam_dynagslam_stack.yaml`
(137), `config/task/maniskill_gs.yaml` (44), `.gitignore`, a root `setup.py` and a symlink under
`third_party/VRL3/`.

**How it was compared.** Every function of the old wrapper was extracted with Python's `ast` module and
diffed (after removing indentation) against its counterpart in the new mapper and wrapper. The old task
yaml and its base were loaded and compared key by key against the composed new config. The old runner was
read against `ManiSkillRunner`. The lists below are the output of that, not a recollection.

## Why the split

`DynaGSLAMWrapper` did three jobs in one `gym.Env`: the SLAM lifecycle, the translation of ManiSkill
observations into DynaGSLAM inputs, and the gym plumbing. On master the plumbing is owned by
`ManiSkillDP3BaseObsWrapper`, and the SLAM lifecycle gets its own home in `baseline_scene_integration/`.
That home exists because a scene mapper has two callers, the online observation wrapper and the offline
dataset converter, so a policy trains on exactly the reconstruction it later sees at rollout. The mapper
therefore takes plain tensors (RGB, depth in meters, intrinsics, camera pose, segmentation, dynamic mask)
and knows nothing about gym, ManiSkill or SAPIEN.

## Where each piece went

| was in `env/maniskill/dynagslam_wrapper.py` | now |
|---|---|
| `SH_C0`, `_quaternion_wxyz_to_matrix` (module level) | module level in `baseline_scene_integration/dynagslam_scene_mapper.py` |
| `__init__`: `slam_args`, `optimization_params`, forcing `use_gt_pose`/`mode`, `gaussian_map`, `_tracker_preprocessor`, `_frame_id` | `DynaGSLAMSceneMapper.__init__` |
| `_init_slam` | `DynaGSLAMSceneMapper.reset()` |
| `_make_dyna_camera(obs, frame_id)` | `DynaGSLAMSceneMapper._make_dyna_camera(rgb, depth_m, intrinsics, cam2world_cv, frame_id)` |
| `_run_slam_step(obs)` | `DynaGSLAMSceneMapper.integrate_frame(rgb, depth_m, intrinsics, cam2world_cv, segmentation, dynamic_mask)` |
| `_extract_gaussians_from_map()` | `DynaGSLAMSceneMapper.get_scene_representation()` |
| `step()`: `global_optimization(select_keyframe_num=-1)` when `done` | **deleted** (change M7) |
| `__init__`: `cam_name`, `num_gaussians`, `use_gsplat_viewer`, `_gaussian_indices`, viewer start | `DynaGSLAMManiSkillDP3Wrapper.__init__` (`env/maniskill/observation_wrapper/dynagslam/maniskill_dynagslam_wrapper.py`) |
| `_build_observation_space(agent_pos_dim)` | `DynaGSLAMManiSkillDP3Wrapper._perception_observation_space()` |
| `_build_obs_dict(raw_obs, force_resample)` + the obs lookups of `_make_dyna_camera` / `_run_slam_step` | `DynaGSLAMManiSkillDP3Wrapper._get_perception_obs_dict(obs, step)` |
| `reset()`: `_init_slam()`, `_gaussian_indices = None` | `DynaGSLAMManiSkillDP3Wrapper._reset_perception_state()`; the rest of `reset()` is the base class |
| `step()`: action cast, flattening of `terminated`/`truncated`/`reward` | base class, same logic |
| `_get_sim_segmentation`, `_subsample_gaussians`, `render`, `_init_gsplat_viewer`, `_update_gsplat_viewer`, `_matrix_to_quaternion_wxyz` | wrapper, same names |
| `env_runner/dynagslam_maniskill_runner.py` | **deleted**; the env construction became the builder function `wrist_cam_dynagslam` in `env_runner/maniskill_env_obs_wrapper_builder.py`, the episode loop is `ManiSkillRunner` |
| `dynagslam:` block of the task yaml | `config/scene_representation/dynagslam.yaml` |
| `config/task/maniskill_gs.yaml` and the rest of the task yaml | **deleted**; the task yaml inherits `maniskill_wrist_cam_gs_base.yaml` |

## Exact changes, file by file

Everything not listed here is character for character the original. Verified identical:
`_quaternion_wxyz_to_matrix`, `_matrix_to_quaternion_wxyz`, `_get_sim_segmentation`,
`_update_gsplat_viewer`, and all 101 keys of the `dynagslam:` config block. The four per-frame debug
`print`s in `_make_dyna_camera` and `integrate_frame` (raw depth, processed depth, vertex finiteness) are
kept on purpose.

### Mapper: `baseline_scene_integration/dynagslam_scene_mapper.py`

- **M1 `__init__`.** Takes `slam_args, optimization_params, control_freq, device="cuda"`. From the old
  wrapper constructor it keeps, unchanged: storing both configs, `slam_args.use_gt_pose = True`,
  `slam_args.mode = "single process"`, `gaussian_map = None`, `_tracker_preprocessor = None`,
  `_frame_id = 0`. New: `self.control_freq` (change M4), `self.device` (change M6),
  `self.last_gs_rgb = None` (change M5).
- **M2 `_init_slam` → `reset()`.** Body unchanged; one line added at the end: `self.last_gs_rgb = None`.
- **M3 `_make_dyna_camera`.** Signature `(obs, frame_id)` → `(rgb, depth_m, intrinsics, cam2world_cv, frame_id)`.
  Inside:
  - `K = cam_params['intrinsic_cv'][0].float()` → `K = intrinsics.float()`
  - `H = rgb.shape[1]; W = rgb.shape[2]` → `H = rgb.shape[0]; W = rgb.shape[1]` (the frame arrives without the env batch dimension)
  - pose: `extr = extrinsic_cv[0]; w2c_4x4 = eye(4); w2c_4x4[:3] = extr; c2w_4x4 = inv(w2c_4x4)` →
    `c2w_4x4 = cam2world_cv.float(); w2c_4x4 = inv(c2w_4x4)`. `R`, `T` and `pose_gt` are computed from these
    exactly as before.
  - `rgb_chw = rgb[0].float().permute(2, 0, 1) / 255.0` → `rgb_chw = rgb.float().permute(2, 0, 1) / 255.0`
  - depth: `depth_m = cam_data['depth'][0].float()`, the `if depth_m.ndim == 3: depth_m = depth_m[..., 0]`
    squeeze and `depth_m = depth_m / 1000.0` → `depth_m = depth_m.float()`. The caller converts to meters
    with the shared `perception_utils.depth_to_meters` (see W4). The `/ 255.0` DynaGSLAM quirk that follows
    is unchanged.
  - `timestamp = frame_id / 20.0` → `timestamp = frame_id / self.control_freq` (change M4)
- **M4 `control_freq`.** Both hardcoded `20.0` timestamps (`_make_dyna_camera`, and `t_old` in
  `integrate_frame`) use `self.control_freq`; the builder passes `base_env.unwrapped.control_freq`, which is
  20 for these envs, so the values are the same.
- **M5 `_run_slam_step` → `integrate_frame`.** Signature `(obs)` →
  `(rgb, depth_m, intrinsics, cam2world_cv, segmentation, dynamic_mask)`. Inside:
  - `seg_mask = self._get_sim_segmentation(obs)` → `seg_mask = dynamic_mask`
  - `frame_map["object_id_map"] = obs["sensor_data"][cam]["segmentation"][0, :, :, 0].to(...)` →
    `frame_map["object_id_map"] = segmentation.to(...)` (same device and dtype cast)
  - a three-line comment ("Pass world-space vertex map back into frame_map ...") removed; no code under it
  - `self._last_gs_rgb = gs_rgb` → `self.last_gs_rgb = gs_rgb`, public because the wrapper's `render()`
    now reads it across the class boundary
- **M6 `_extract_gaussians_from_map` → `get_scene_representation`.**
  - empty-map fallback: `torch.device('cuda')` → `torch.device(self.device)`; a TODO added noting the
    one-row fallback is smaller than `num_gaussians`. (This branch cannot trigger online: `reset()` integrates
    the first frame before the first observation.)
  - added output key `gs_surface_normals = gp['normal']` (also a zero row in the fallback).
    `shape_meta` of the gs task and `GSplatDP3`'s encoder carry per-Gaussian surface normals, which the old
    wrapper never produced, so a GSplatDP3 policy could not have consumed its observations. DynaGSLAM defines
    them as each Gaussian's smallest-scale axis (`GaussianPointCloud.get_normal`). All other keys unchanged.
- **M7 deleted: global optimisation after `done`.** The old `step()` ran
  `self.gaussian_map.global_optimization(copy.deepcopy(self.optimization_params), select_keyframe_num=-1)`
  when the episode ended, DynaGSLAM's all-keyframes final polish. After `done` no action is taken and
  `reset()` discards the map, so it cost one full optimisation per episode for nothing the policy does or is
  scored on; we care about policy performance, not a perfect final reconstruction. The in-episode schedule
  inside `Mapping.mapping()` is untouched (local optimisation every `gaussian_update_frame` frames, global
  over `global_keyframe_num` keyframes on keyframes). `import copy` went with it. `final_global_iter: 10`
  stays in the config: `Mapping.__init__` reads it unconditionally and would raise without it, although only
  the deleted path consumed it.
- **M8 imports.** `fov2focal` was imported and never used; dropped. The DynaGSLAM imports (`Camera`,
  `Mapping`, `Tracker`, `move_to_gpu`, `move_to_cpu`) moved here from the wrapper.

### Wrapper: `env/maniskill/observation_wrapper/dynagslam/maniskill_dynagslam_wrapper.py`

- **W1 `__init__`.** Signature `(env, slam_args, optimization_params, cam_name="right_cam", num_gaussians=1024, use_gsplat_viewer=False)`
  → `(env, representation_space, agent_proprio_dim, cam_name, scene_mapper, num_gaussians, min_opacity, use_gsplat_viewer=False)`.
  Kept: `cam_name`, `num_gaussians`, `use_gsplat_viewer`, `_gaussian_indices = None`, `_init_gsplat_viewer()`
  when the viewer is on. Moved to the mapper: everything in M1. Removed because the base class owns it:
  the action-space cast to `gym.spaces.Box`, the observation-space build from `qpos` dim, `_last_rgb`.
  Removed as dead: `_env_got_reset` (set in `reset()`, never read), class attribute `DYNAMIC_SEG_IDS`
  (never read). New: `min_opacity` (W5), `scene_mapper` injected by the builder.
- **W2 `_build_observation_space(agent_pos_dim)` → `_perception_observation_space()`.** Returns a plain dict
  of the gs_* Boxes instead of assigning `self.observation_space`; the `agent_proprio` Box is gone (the base
  adds it); a `gs_surface_normals` Box `(num_gaussians, 3)` in `[-1, 1]` is added (M6). The six other Boxes
  are unchanged.
- **W3 `_build_obs_dict(raw_obs, force_resample=False)` → `_get_perception_obs_dict(obs, step)`.**
  - new at the top: the obs lookups that used to sit inside `_make_dyna_camera` / `_run_slam_step`:
    `rgb = cam_data['rgb'][0]`, `depth_m = depth_to_meters(cam_data['depth'][0, ..., 0])`,
    `segmentation = cam_data['segmentation'][0, :, :, 0]`, `intrinsics = cam_params['intrinsic_cv'][0]`,
    `cam2world_cv = cam2world_cv_from_extrinsic_cv(cam_params['extrinsic_cv'][0])`,
    `dynamic_mask = self._get_sim_segmentation(obs)`; then `self.scene_mapper.integrate_frame(...)`.
    Before, `_run_slam_step(raw_obs)` was called by `reset()`/`step()` right before `_build_obs_dict`; the
    order "integrate, then read the map, then subsample" is the same.
  - `full_gsplat_data = self._extract_gaussians_from_map()` → `self.scene_mapper.get_scene_representation()`
  - removed: the `qpos` proprio block and the `_last_rgb` caching (base class, see R3 for the consequence)
  - `self._subsample_gaussians(full_gsplat_data, force_resample)` →
    `self._subsample_gaussians(full_gsplat_data, force_resample=(step == 0))`. `step` is the env's
    `elapsed_steps`: 0 on the observation produced by `reset()`, positive on every `step()`. That reproduces
    the old two call sites (`True` from `reset()`, `False` from `step()`). A TODO marks that the resampling is
    still triggered twice on reset, by this flag and by `_gaussian_indices = None`, exactly as in the original.
- **W4 depth in meters.** The old docstring said ManiSkill returns meters, the old code divided by 1000.
  `depth_to_meters` converts the int16 millimeters every depth source here delivers and is what the other
  wrappers use too. Negative depth and depth beyond `MAX_VALID_DEPTH_METERS` become 0, invalid; ManiSkill
  saturates int16 at 32767 for pixels without geometry, so those are invalid as well.
- **W5 `_subsample_gaussians`.** Two lines: `high_mask = opacities >= 0.98` → `>= self.min_opacity`, and the
  same in the docstring. `min_opacity` comes from `task.min_opacity` (0.98 in `maniskill_wrist_cam_gs_base`),
  so the value is the same. Everything else in the method is unchanged.
- **W6 `_reset_perception_state()`** (new hook, called by the base at the start of `reset()`):
  `self.scene_mapper.reset()` and `self._gaussian_indices = None`. The old `reset()` did
  `env.reset()`, `_init_slam()`, `_gaussian_indices = None`, `_run_slam_step()`, `_build_obs_dict(force_resample=True)`;
  the base now does `_reset_perception_state()`, `env.reset()`, then W3 with `step == 0`. SLAM init and
  `env.reset()` swapped order; they do not depend on each other.
- **W7 `render()`.** `self._last_rgb` → `super().render(mode)` (the base caches the same camera's RGB as
  uint8 numpy); `self._last_gs_rgb` → `self.scene_mapper.last_gs_rgb`. Fallback before any frame: the old
  `np.zeros((256, 512, 3))` → the base's `(256, 256, 3)`. The side-by-side (ground truth left,
  reconstruction right) is unchanged.
- **W8 `_init_gsplat_viewer`.** `device = torch.device('cuda')` → `device = self.device`.
- **W9 imports.** `gym` (only `spaces` is used), `copy`, `fov2focal`, DynaGSLAM internals removed; added
  `depth_to_meters`, `cam2world_cv_from_extrinsic_cv`, `ManiSkillDP3BaseObsWrapper`.

### Runner: `env_runner/dynagslam_maniskill_runner.py` → `ManiSkillRunner` + builder

The old runner was deleted, so this is a comparison of what happens at rollout, old vs new:

| | old `DynaGSLAMManiSkillRunner` | new `ManiSkillRunner` + `wrist_cam_dynagslam` |
|---|---|---|
| R1 robot and camera | `robot_uids="fr3_umi_wrist435_modified"` hardcoded. At his GSWorld pin (`b32c4a1`) that robot carried the wrist camera as a RealSense D435i twin: 640×480, fx ≈ 606 (≈ 56° horizontal FOV), mounted at `camera_link`; 16 robot links, so SAPIEN segmentation ids 1–16 were the robot | `fr3_umi_wrist_zedxmini` from `maniskill_base.yaml` (C6), the only robot the wrist-cam envs still support. Its wrist camera is the ZED X Mini twin: 960×600, fx ≈ 370 (≈ 105° horizontal FOV), mounted at `zed_left_camera_frame`; 17 robot links (ids 1–17; id 17 `zed_left_camera_frame` has no visual mesh), table 18, ground 19, objects 20+. DynaGSLAM's hardcoded `robot_mask = 1..16` in `mapping()` still covers every visible robot link. **Every frame DynaGSLAM integrates therefore differs from before: 1.9× the pixels under the same `uniform_sample_num: 50000`, a much wider view, a slightly different viewpoint.** Depth is int16 millimeters in both, so the meter conversion is unchanged |
| R2 control | `control_mode="pd_joint_pos"` hardcoded; `representation_space` was received and ignored (conditional commented out) | `pd_ee_pos_quat` + `RelativeEEControlWrapper`, because `representation_space` is `relative_ee_pose`; `pd_joint_pos` for `abs_joint_pos` |
| R3 `agent_proprio` | joint positions `qpos` (from the wrapper) | for `relative_ee_pose`: TCP pose + gripper state, 11 values, from the base wrapper; matches the `shape_meta` (11) that his config already declared via `maniskill_base` |
| R4 policy input | exactly `gs_positions, gs_rotations_9d, gs_log_scales, gs_opacities, gs_rgb, agent_proprio` | every obs key (so also `gs_surface_normals`) plus alias `point_cloud → gs_positions` |
| R5 sim | `sim_backend="gpu"`, `sim_config(100, 20)`, `obs_mode="rgb+depth+segmentation"` | same values, from `device` and the yaml chain |
| R6 horizon | `eval_episodes 20`, `max_steps 400`, `n_obs_steps 2`, `n_action_steps 8` (from `maniskill_base` at his branch point, overriding his runner defaults 1000/8) | same values |
| R7 video | saved by hand with `imageio` at fps 10 to `eval_videos/{prefix}_ep_{i}.mp4` | `VideoSavingCallback` fps 10 to `eval_videos/{prefix}_ep_{i}_{suffix}.mp4` plus `TrajectoryPlotCallback` (C7); the runner's `AttentionOverlayWrapper` sits in the stack but is a pass-through without `GaussianAttentionPlotCallback` |
| R8 initial states | random training episode from `dataset.replay_buffer` (zarr) | random training episode via `dataset.get_episode_init_data` (memmap) |
| R9 wrapper construction | `DynaGSLAMWrapper(env, slam_args, optimization_params, cam_name, num_gaussians, use_gsplat_viewer)` with `optimization_params = slam_args` | builder: `slam_args = OmegaConf.create(to_container(scene_representation))`, `DynaGSLAMSceneMapper(slam_args, optimization_params=slam_args, control_freq, device)`, then the wrapper. Same one-config-for-both as before |
| R10 `__main__` | smoke test driving the env with `CanStackMotionPlanningPolicy` | gone with the file |
| R11 logging | mean reward, mean success | plus `steps_p25/median/p75` to success |

### Config

- **C1 `dynagslam:` block → `config/scene_representation/dynagslam.yaml`.** 101 keys, every value identical
  (checked programmatically). It is composed via the defaults list of the new launch config
  `config/wrist_cam_dynagslam_gsplat_dp3.yaml` and reaches the builder as `scene_representation`
  (was `dynagslam_config: ${task.dynagslam}`).
- **C2 `config/task/maniskill_gs.yaml` deleted.** Its `shape_meta` (7 keys, shapes and types) and
  `num_gaussians: 1024` are identical to `maniskill_wrist_cam_gs_base.yaml`, which the task now inherits.
  Inherited on top, which his config did not have: `min_opacity: 0.98`, `env_runner.obs_key_aliases`, the
  dataset parameters shared by every Gaussian source, `train/val_data_augmentations: []`. The dataset class
  and paths are `???` in the base and the DynaGSLAM leaf leaves them unset (a TODO): his config had no
  dataset either, so the task is rollout-only until a DynaGSLAM dataset exists or the gsplat paths are
  filled in.
- **C3 `env_runner` block of the task yaml.** `_target_` is now `ManiSkillRunner` (from `maniskill_base`);
  `num_gaussians`, `use_gsplat_viewer`, `cam_name` moved into `env_runner.maniskill_env_obs_wrapper_builder`
  together with `representation_space`, `agent_proprio_dim`, `scene_representation`, `min_opacity`.
- **C4 `use_gsplat_viewer: True` → `false`.** The only value of his task yaml that changed (the robot and control mode were hardcoded in his runner, see R1, R2). A viser server per
  rollout is not wanted during training-time evaluation; set it to `true` for inspecting the map.
- **C5 gs task split (master change).** `maniskill_wrist_cam_gsworld_stack.yaml` held the gs_* contract
  together with the GSWorld builder; hydra merges parent and child nodes, so a DynaGSLAM leaf inheriting it
  would have passed `scene_gs_cfg_name` to a builder without that parameter. The contract now lives in
  `maniskill_wrist_cam_gs_base.yaml` with `???` for the obs-wrapper builder, the callbacks and the dataset
  class and paths; each leaf fills those. The GSWorld leaf was renamed from `…_gs_stack` to
  `…_gsworld_stack` (launch configs, script comments and README updated). The composed `wrist_cam_gsplat_dp3`
  task is identical before and after except for `name` (checked field by field).
- **C6 `robot_uids: fr3_umi_wrist_zedxmini`** in `maniskill_base.yaml`, replacing `fr3_umi_wrist435_modified`,
  which predates the ZED migration (gsplat_policy `03bff4c`, 2026-08-28) and is no registered agent any more.
  Every maniskill task in the repo is a wrist-cam task and the wrist-cam envs support only the ZED robot, and
  the base's own comment says leaves override only `task_name`, the builder and the callbacks, so the robot
  belongs in the base. This also affected gsplat rollouts on master.
- **C7 callbacks of the DynaGSLAM task.** The gs base's `GaussianAttentionPlotCallback` recolors GSWorld's
  scene through the runner's `AttentionOverlayWrapper` (`overwrite_gs_*` on the `WristCamGSWorldWrapper`);
  with DynaGSLAM there is no such scene and the first `step()` after a policy call fails with
  `'StackFr3WristCamSimpleEnv' object has no attribute 'moving_gaussians'`. The DynaGSLAM leaf therefore
  declares `VideoSavingCallback` (fps 10, no attention legend) and `TrajectoryPlotCallback`, exactly as the
  point-cloud leaf does. Without the attention callback the overlay wrapper is a pass-through.

### Untouched from the branch

`.gitignore` (`*.mp4`), the root `setup.py` (a duplicate of `3D-Diffusion-Policy/setup.py`) and the symlink
`third_party/VRL3/src/diffusion_policy_3d` (points into `/home/fawad/...`). The last two should go in a
separate commit.

## Known issues in the original logic, not changed by the port

Found while checking the port; the port reproduces the original behaviour on purpose, fixing is a separate step.

1. **Stored Gaussian indices do not follow Gaussians.** `_subsample_gaussians` draws indices once per
   episode and reuses them. DynaGSLAM's `global_params` is rebuilt every call as concat[unstable static,
   stable static, dynamic]; every sub-cloud is compacted on deletion and Gaussians move between the static
   clouds, so an index names a slot, not a Gaussian. The `index < N` guard never fires because the map only
   grows. Measured over reset + 24 hold-pose control steps: indices identical throughout, yet 40 / 28 / 17 %
   of the slots changed occupant (position jump > 5 cm) in steps 1–3, and from step 4 on 100 % of the slots
   lay in the unstable static cloud while the dynamic cloud (robot links + cubes) was 22–24 % of the map.
2. **Static vs dynamic.** The GSWorld observation and the GS training data contain only the moving
   Gaussians (robot links and movable actors); the DynaGSLAM observation is dominated by the table. Sampling
   from the dynamic cloud only (`Mapping.dyna_params`, or the last `get_dyna_num` rows) would match.
3. **The opacity filter is a no-op:** `init_opacity: 0.99` ≥ `min_opacity: 0.98`.
4. **A view without any mapped Gaussian fails inside DynaGSLAM's rasterizer.** `Mapping.temp_points_init`
   renders the current map from the new camera; when no mapped Gaussian falls into the view (the debug line
   reads `visible Gaussians: 0 / N`), the rasterizer launches with zero blocks and raises
   `CUDA error: invalid configuration argument`. The runner's random dummy policy provokes it within about
   20 policy calls by swinging the camera to the horizon; a policy early in training can do the same.

## Files

New: `baseline_scene_integration/dynagslam_scene_mapper.py`,
`env/maniskill/observation_wrapper/dynagslam/{__init__,maniskill_dynagslam_wrapper}.py`,
`env_runner/maniskill_env_obs_wrapper_builder.py::wrist_cam_dynagslam`, `config/scene_representation/dynagslam.yaml`,
`config/task/maniskill_wrist_cam_gs_base.yaml` (C5), `config/wrist_cam_dynagslam_gsplat_dp3.yaml` (the launch config).

Rewritten: `config/task/maniskill_wrist_cam_dynagslam_stack.yaml`, `config/task/maniskill_wrist_cam_gsworld_stack.yaml` (C5),
`config/task/maniskill_base.yaml` (C6).

Deleted: `env/maniskill/dynagslam_wrapper.py`, `env_runner/dynagslam_maniskill_runner.py`, `config/task/maniskill_gs.yaml`.

## Verified

In `gsplat_policy_dynagslam`, with DynaGSLAM on `dynagslam_integration`: the config composes and the builder
block's keys equal the builder's parameters; `wrist_cam_gsplat_dp3`'s task composes identically to before the
split; `ManiSkillRunner` instantiates (SAPIEN env + `Mapping`, 4.4 s); `reset()` integrates the first frame in
1.5 s and returns all seven gs_* / proprio keys at `(n_obs_steps, 1024, ·)`; three hold-pose policy steps
(8 control steps each) take 5.7 / 7.3 / 8.3 s, i.e. DynaGSLAM costs roughly 0.7–1 s per control step and grows
with the map (155k Gaussians after 25 frames). Positions are finite, normals have unit length, every returned
opacity is ≥ `min_opacity`, and `render()` gives the 600×1920 GT | reconstruction pair.

Full `ManiSkillRunner.run()` loop with a dummy policy (small random relative EE motions), one episode of 40
control steps (5 policy calls) each on `StackFr3WristCamSimpleEnv-v1` and `StackFr3WristCamEnv-v1`, both with
`fr3_umi_wrist_zedxmini`: 40.4 s and 41.3 s per episode, no callback failure, `eval_videos/*.mp4` with 41
frames of 600×1920 (GT | reconstruction) and `eval_plots/*_tcp_state_time.png` written. Before C7 the same
run failed on the inherited attention callback. Not exercised: a trained GSplatDP3 checkpoint and episodes
to `done` by success.

## Running it

Use the `gsplat_policy_dynagslam` conda env. It is the only env with DynaGSLAM's vendored CUDA rasterizer
(`diff_gaussian_rasterization_depth`, built in place from
`submodules/DynaGSLAM_official/submodules/diff-gaussian-rasterizer-depth/`) and it carries the same editable
installs of `diffusion_policy_3d`, `gsplat_envs`, `gsworld` and `dynagslam` as `gsplat_policy`. Importing the
mapper in `gsplat_policy` fails on that rasterizer.

`submodules/DynaGSLAM_official` must be on `dynagslam_integration`: the wrapper passes
`frame_map["object_id_map"]`, which only that branch's `Mapping` consumes.

```bash
conda activate gsplat_policy_dynagslam
bash scripts/train_policy.sh wrist_cam_dynagslam_gsplat_dp3 maniskill_wrist_cam_dynagslam_stack <run_label> <seed> <gpu>
```

Smoke tests without a checkpoint go through the runner's `__main__`, whose default config is this task.
The random dummy policy, and Fawad's motion-planning oracle from `gsplat_envs`, which emits joint targets
and therefore needs the joint-space representation:

```bash
python -m diffusion_policy_3d.env_runner.maniskill_runner task.env_runner.eval_episodes=1
python -m diffusion_policy_3d.env_runner.maniskill_runner task.representation_space=abs_joint_pos +test_policy=motion_planning task.env_runner.eval_episodes=1
```

The oracle stacks the can in about 200 control steps (one episode takes about five minutes with DynaGSLAM)
and writes `test_eval_output/eval_videos/*_success.mp4` next to the joint-position plots.

The DynaGSLAM task has no dataset yet (`???` in the base, TODO in the leaf), so `train_policy.sh` stops at the
dataset; for a training run on the gsplat data, copy the `dataset` block of the GSWorld leaf into the DynaGSLAM
leaf. Rollouts observe DynaGSLAM's reconstruction. DynaGSLAM prints per frame (`camera-space z`, `visible Gaussians`, `map update`
progress bars) on top of the four debug prints kept from the wrapper.
