""" Perception wrapper for `ManiSkillRunner`.

Each builder is a plain function that applies the perception-specific wrappers
(Gaussian-splatting / point-cloud) on top of a base ManiSkill env. The config selects a
builder by `_target_` (as a hydra `_partial_`) and binds its perception-specific params;
`ManiSkillRunner` injects the shared runtime params (`base_env`, `device`, `representation_space`) when it calls the partial.

Adding a new use case = adding a new builder here + a config pointing at it.
"""

import argparse

# NOTE: importing the gsworld wrappers appends GS_DIR to sys.path, which is what makes the
# top-level `arguments` module importable — so this import MUST precede `from arguments ...`.
from gsworld.mani_skill.utils.wrappers import WristCamGSWorldWrapper
from arguments import PipelineParams

from diffusion_policy_3d.env.maniskill.observation_wrapper.gaussian_splatting.maniskill_wrist_cam_gs_wrapper import WristCamGSManiskillDP3Wrapper
from diffusion_policy_3d.env.maniskill.observation_wrapper.pcd.maniskill_pcd_wrapper import SingleStepPCDManiSkillDP3Wrapper
from diffusion_policy_3d.env.maniskill.observation_wrapper.pcd.maniskill_pcd_spatial_memory_wrapper import SpatialMemoryPCDManiSkillDP3Wrapper


def wrist_cam_online_gaussian_splatting(
    *,
    # injected by ManiSkillRunner
    base_env,
    device,
    # defined in config, injected by hydra
    representation_space,
    agent_proprio_dim,
    n_obs_steps,
    n_action_steps,
    scene_gs_cfg_name,
    num_gaussians,
    min_opacity,
    use_gsplat_viewer,
):
    """
    Wrist-camera online Gaussian-splatting perception head.
    """
    assert representation_space in ("abs_joint_pos", "relative_ee_pose"), \
        f"representation_space must be 'abs_joint_pos' or 'relative_ee_pose', got '{representation_space}'"

    robot_pipe = PipelineParams(argparse.ArgumentParser())
    env = WristCamGSWorldWrapper(
        base_env,
        robot_pipe,
        scene_gs_cfg_name=scene_gs_cfg_name,
        device=device,
        use_gsplat_viewer=use_gsplat_viewer,
    )
    env = WristCamGSManiskillDP3Wrapper(
        env,
        representation_space,
        agent_proprio_dim=agent_proprio_dim,
        num_gaussians=num_gaussians,
        min_opacity=min_opacity,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
    )
    return env


def pcd(
    *,
    # injected by ManiSkillRunner
    base_env,
    device,
    # defined in config, injected by hydra
    representation_space,
    agent_proprio_dim,
    cam_name,
    num_points,
):
    """
    Point-cloud perception head.

    NOTE: relative_ee_pose is not yet supported for point clouds — RelativeEEControlWrapper
    only transforms gs_* keys, not 'point_cloud'.
    """
    if representation_space == "relative_ee_pose":
        raise NotImplementedError(
            "relative_ee_pose is not yet implemented for the point-cloud representation: "
            "RelativeEEControlWrapper only transforms gs_* keys, not 'point_cloud'."
        )

    env = SingleStepPCDManiSkillDP3Wrapper(
        base_env,
        representation_space,
        agent_proprio_dim=agent_proprio_dim,
        cam_name=cam_name,
        num_points=num_points,
    )
    return env


def spatial_memory_pcd(
    *,
    # injected by ManiSkillRunner
    base_env,
    device,
    # defined in config, injected by hydra
    representation_space,
    agent_proprio_dim,
    cam_name,
    scene_representation,
    zarr_path,
):
    """
    Spatial memory point-cloud perception head.

    Mapper params come from the composed scene_representation config
    (config/scene_representation/spatial_memory_pcd.yaml -- the same file the
    dataset converter loaded and stamped into the zarr). We ASSERT the composed
    config equals that stamp: the dataset was recorded with these values (e.g.
    voxel size), so evaluating with different ones would be a silently invalid
    comparison.

    NOTE: relative_ee_pose is not yet supported for point clouds — RelativeEEControlWrapper
    only transforms gs_* keys, not 'point_cloud'.
    """
    if representation_space == "relative_ee_pose":
        raise NotImplementedError(
            "relative_ee_pose is not yet implemented for the point-cloud representation: "
            "RelativeEEControlWrapper only transforms gs_* keys, not 'point_cloud'."
        )

    import json

    import zarr
    from omegaconf import OmegaConf

    from diffusion_policy_3d.baseline_scene_integration.spatial_memory_pcd_scene_mapper import (
        SpatialMemoryPcdSceneMapper,
    )

    from diffusion_policy_3d.dataset.multi_zarr_dataset import discover_zarr_paths

    # tripwire: composed config must equal what EVERY dataset of the training run was
    # converted with (json round-trip normalizes container types on both sides)
    composed = json.loads(json.dumps(OmegaConf.to_container(scene_representation, resolve=True)))
    for path in discover_zarr_paths(zarr_path):
        stored_attrs = dict(zarr.open_group(path, mode="r")["meta"].attrs)
        assert "scene_representation" in stored_attrs, (
            f"{path} lacks meta.attrs scene_representation -- reconvert with the "
            "current convert_wrist_cam_gsworld_to_spatial_memory_pcd.py."
        )
        stored = json.loads(json.dumps(stored_attrs["scene_representation"]))
        assert composed == stored, (
            "scene_representation config does not match the one the dataset was "
            f"converted with.\n  composed (config/scene_representation/*.yaml): {composed}\n"
            f"  stored (zarr meta.attrs of {path}): {stored}\n"
            "The dataset was recorded with the stored values -- evaluating with "
            "different ones is invalid. Revert the yaml or reconvert the dataset."
        )

    scene_mapper = SpatialMemoryPcdSceneMapper(
        voxel_size=composed["voxel_size"],
        eliminate_background=composed["eliminate_background"],
        device=device,
    )

    env = SpatialMemoryPCDManiSkillDP3Wrapper(
        base_env,
        representation_space,
        agent_proprio_dim=agent_proprio_dim,
        cam_name=cam_name,
        scene_mapper=scene_mapper,
        # num_points is not a mapper parameter: the mapper accumulates the full memory and the
        # wrapper samples this many points out of it per step, with the same shared function
        # the PointCloudFPS augmentation applies to every training batch.
        num_points=composed["num_points"],
    )
    return env

def wrist_cam_nvblox_reconstruction(
    *,
    # injected by ManiSkillRunner
    base_env,
    device,
    # defined in config, injected by hydra
    representation_space,
    agent_proprio_dim,
    n_obs_steps,
    n_action_steps,
    cam_name,
    scene_representation,
    zarr_path,
    use_mock_scene_mapper=False,
):
    """
    Wrist-camera nvblox spatial-memory perception head (mindmap baseline).

    Chosen mapper params come from the composed scene_representation config
    (config/scene_representation/nvblox.yaml -- the same file the dataset
    converter loaded), including num_vertices, the cloud size the policy
    consumes (the live mapper samples it directly from the mesh -- the same
    distribution as the dataset's stored-superset -> num_vertices two-stage
    sampling). Data-derived values (workspace_bounds, embedding_dim) come from
    the dataset at zarr_path. The converter stamped its effective params into
    the zarr, and we ASSERT the composed config equals that stamp: the dataset
    was recorded with these values (e.g. voxel size), so evaluating with
    different ones would be a silently invalid comparison.

    The nvblox scene mapper (TSDF + feature grid inside, featurized vertex cloud
    out) requires nvblox_torch + the RADIO extractor -- available in the
    mindmap_baseline container, not on the host. `use_mock_scene_mapper=True`
    substitutes a random-cloud mock so the env/runner plumbing can be tested
    anywhere (imports are lazy for the same reason).
    """
    import json

    from omegaconf import OmegaConf

    from diffusion_policy_3d.baseline_scene_integration.nvblox_scene_mapper import (
        read_dataset_perception_facts,
    )
    from diffusion_policy_3d.env.maniskill.observation_wrapper.nvblox.maniskill_wrist_cam_nvblox_wrapper import (
        MockNvbloxSceneMapper,
        WristCamNvbloxManiskillDP3Wrapper,
    )

    facts = read_dataset_perception_facts(zarr_path)

    # tripwire: composed config must equal what the dataset was converted with
    # (json round-trip normalizes container types on both sides)
    composed = json.loads(json.dumps(OmegaConf.to_container(scene_representation, resolve=True)))
    stored = json.loads(json.dumps(facts["scene_representation"]))
    assert composed == stored, (
        "scene_representation config does not match the one the dataset was "
        f"converted with.\n  composed (config/scene_representation/*.yaml): {composed}\n"
        f"  stored (zarr meta.attrs of {zarr_path}): {stored}\n"
        "The dataset was recorded with the stored values -- evaluating with "
        "different ones is invalid. Revert the yaml or reconvert the dataset."
    )

    if use_mock_scene_mapper:
        scene_mapper = MockNvbloxSceneMapper(
            workspace_bounds=facts["workspace_bounds"],
            # Stands in for the mesh, whose size the stored superset approximates.
            num_mesh_vertices=facts["num_stored_vertices"],
            embedding_dim=facts["embedding_dim"],
            device=device,
        )
    else:
        from diffusion_policy_3d.baseline_scene_integration.nvblox_scene_mapper import (
            NvbloxSceneMapper,
            NvbloxSceneMapperConfig,
        )

        mapper_config = NvbloxSceneMapperConfig(
            **composed["mapper_config"],
            aabb_min_m=list(facts["workspace_bounds"][0]),
            aabb_max_m=list(facts["workspace_bounds"][1]),
        )
        scene_mapper = NvbloxSceneMapper(
            mapper_config,
            feature_type=composed["feature_type"],
            device=device,
        )

    return WristCamNvbloxManiskillDP3Wrapper(
        env=base_env,
        representation_space=representation_space,
        agent_proprio_dim=agent_proprio_dim,
        cam_name=cam_name,
        scene_mapper=scene_mapper,
        # Not mapper parameters: the mapper returns the whole mesh and the wrapper samples
        # this many vertices out of it per step, exactly as the dataset does per access.
        num_vertices=composed["num_vertices"],
        vertex_sampling_method=composed["vertex_sampling_method"],
    )
