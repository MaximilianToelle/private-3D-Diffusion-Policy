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
from diffusion_policy_3d.env.maniskill.observation_wrapper.pcd.maniskill_pcd_wrapper import PCDManiSkillDP3Wrapper


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


# def pcd(
#     *,
#     # injected by ManiSkillRunner
#     base_env,
#     device,
#     # defined in config, injected by hydra
#     representation_space,
#     agent_proprio_dim,
#     scene_gs_cfg_name,
#     cam_name,
#     num_points,
# ):
#     """
#     Point-cloud perception head.

#     NOTE: relative_ee_pose is not yet supported for point clouds — RelativeEEControlWrapper
#     only transforms gs_* keys, not 'point_cloud'.
#     TODO: still using GSWorldWrapper due to hardcoded GS cfg -> parameterize in yaml
#     """
#     if representation_space == "relative_ee_pose":
#         raise NotImplementedError(
#             "relative_ee_pose is not yet implemented for the point-cloud representation: "
#             "RelativeEEControlWrapper only transforms gs_* keys, not 'point_cloud'."
#         )

#     robot_pipe = PipelineParams(argparse.ArgumentParser())
#     env = GSWorldWrapper(
#         base_env,
#         robot_pipe,
#         scene_gs_cfg_name=scene_gs_cfg_name,
#         device=device,
#     )
#     env = PCDManiSkillDP3Wrapper(
#         env,
#         representation_space,
#         agent_proprio_dim=agent_proprio_dim,
#         cam_name=cam_name,
#         num_points=num_points,
#     )
#     return env