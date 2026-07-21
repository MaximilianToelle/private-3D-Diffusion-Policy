"""Evaluation callbacks for `ManiSkillRunner`.

Logging and visualization are factored out of the eval loop into callbacks so the
runner itself stays clean. Which callbacks are active is decided in
config (`task.env_runner.callbacks`).

Hook order per `ManiSkillRunner.run`:
    at_run_start                -> once, before any episode
      before_episode_start_after_env_reset          -> per episode
        after_action_predict              -> per step, BEFORE env.step
        after_env_step                 -> per step
      at_episode_end            -> per episode (guarded: exceptions are logged, not raised)
    at_run_end                  -> once, after all episodes (guarded)

`after_action_predict` / `after_env_step` run in the hot loop and are intentionally *not* guarded, so
misconfigured callbacks fail loudly.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import cv2
import imageio
from scipy.spatial.transform import Rotation
from termcolor import cprint

from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.env.maniskill.relative_ee_control_wrapper import RelativeEEControlWrapper
from diffusion_policy_3d.env.maniskill.attention_overlay_wrapper import AttentionOverlayWrapper


def _find_wrapper(env, wrapper_cls):
    """Walk the gym wrapper chain and return the first wrapper of `wrapper_cls`.

    Unlike `env.unwrapped` (which skips straight to the base env at the bottom of the
    stack), this stops at a specific *intermediate* wrapper. Callbacks read state off
    such wrappers; searching by type keeps them robust to changes in wrapper ordering,
    instead of hardcoding a fixed `.env.env...` descent.

    Raises `LookupError` (fail-fast) if the wrapper is absent!
    """
    node = env
    while node is not None:
        if isinstance(node, wrapper_cls):
            return node
        node = getattr(node, "env", None)
    raise LookupError(
        f"{wrapper_cls.__name__} not found in the env wrapper chain -- this callback "
        f"requires it. Check `task.env_runner` wraps the env with {wrapper_cls.__name__}."
    )


def _find_env_attr(env, attr: str):
    """Walk the env chain (descending through both `.env` and `.unwrapped`) and return the
    first object that defines `attr`.

    Used for attributes like `control_freq` that live on the ManiSkill base env rather than
    on a nameable wrapper class, so `_find_wrapper` (which matches by type) does not apply.
    """
    seen = set()
    node = env
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        # A node "defines" the attr if it's in the instance __dict__ OR is a real class
        # member (e.g. a @property like `control_freq`, which lives in the type, not the
        # instance dict). We check the MRO explicitly rather than using `hasattr`, because
        # gym wrappers delegate unknown attributes to the inner env via `__getattr__`, so
        # `hasattr` would spuriously match at every wrapper level.
        if attr in getattr(node, "__dict__", {}) or any(
            attr in vars(base) for base in type(node).__mro__
        ):
            return getattr(node, attr)
        nxt = getattr(node, "env", None)
        if nxt is None or nxt is node:
            unwrapped = getattr(node, "unwrapped", None)
            nxt = unwrapped if unwrapped is not node else None
        node = nxt
    raise LookupError(f"No object exposing '{attr}' found in the env chain.")


class EvalCallback:
    """No-op base class. Override the hooks you need.

    Callbacks are standalone observers: each reads state and produces its own artifact,
    with no ordering dependency on another callback. State that must be *mutated* for
    other components to see (e.g. recoloring the Gaussian scene so the video shows policy
    attention) belongs in an env wrapper, not here -- see `AttentionOverlayWrapper`.
    """

    def at_run_start(self, runner, policy):
        pass

    def before_episode_start_after_env_reset(self, runner, *, episode_idx, init_state, expert_trajectory):
        pass

    def after_action_predict(self, runner, *, obs_dict, action, action_dict, policy, env):
        pass

    def after_env_step(self, runner, *, obs, action, reward, done, info, policy, env):
        pass

    def at_episode_end(self, runner, *, episode_idx, is_success, prefix, env, expert_trajectory, info):
        pass

    def at_run_end(self, runner, *, prefix, log_data):
        pass


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------
class VideoSavingCallback(EvalCallback):
    """Consumes the frames captured by `SimpleVideoRecordingWrapper` and saves an mp4
    per episode.

    Producer/consumer split: the wrapper captures a frame on every
    step; this callback is the consumer that turns the buffer into
    an artifact -- naming by success/failure, choosing the output dir, optionally
    stitching on the attention legend. It must run before the next `env.reset()` clears
    the buffer, which is guaranteed because it lives in `at_episode_end`.

    Requires `SimpleVideoRecordingWrapper` in the env stack (enforced via `_find_wrapper`).

    Args:
        fps: playback frame rate of the saved video.
        add_attention_legend: append the Gaussian-attention colorbar legend below the
            frames. Only meaningful together with `GaussianAttentionPlotCallback`.
        colormap: matplotlib colormap name for the legend.
    """

    def __init__(self, fps: int = 10, add_attention_legend: bool = False, colormap: str = 'cool'):
        self.fps = fps
        self.add_attention_legend = add_attention_legend
        self.colormap = colormap

    def at_episode_end(self, runner, *, episode_idx, is_success, prefix, env, expert_trajectory, info):
        video = _find_wrapper(env, SimpleVideoRecordingWrapper).get_video()  # (T, C, H, W)

        video_dir = os.path.join(runner.output_dir, "eval_videos")
        os.makedirs(video_dir, exist_ok=True)
        video_to_save = video.transpose(0, 2, 3, 1)  # (T, H, W, C)
        if self.add_attention_legend:
            video_to_save = add_legend_to_video(video_to_save, colormap=self.colormap)

        suffix = "success" if is_success else "failure"
        out_path = os.path.join(video_dir, f"{prefix}_ep_{episode_idx}_{suffix}.mp4")
        imageio.mimsave(out_path, video_to_save, fps=self.fps, macro_block_size=1)
        cprint(f"Saved evaluation video to {out_path}", "cyan")


# ---------------------------------------------------------------------------
# Gaussian attention heatmap + unique-Gaussian tracking
# ---------------------------------------------------------------------------
class GaussianAttentionPlotCallback(EvalCallback):
    """Observes the policy's per-inference max-pool attention: plots the number of unique
    selected Gaussians over time, and feeds the attention indices to the
    `AttentionOverlayWrapper` so the recorded video shows which Gaussians are attended to.

    Pure observer: it does NOT mutate the scene itself -- the recoloring lives in
    `AttentionOverlayWrapper` (an env wrapper). This callback only reads the policy and
    pushes indices to that wrapper.

    Requires a GSplat policy exposing `policy.obs_encoder.extractor._latest_pool_indices`
    and an `AttentionOverlayWrapper` in the env stack. Both are checked fail-fast in
    `at_run_start`. Enable only for the Gaussian-splatting use case.
    """

    def __init__(self):
        self.num_unique_gaussians_over_all_episodes: List[np.ndarray] = []
        self.num_unique_gaussians_over_episode: List[int] = []

    def at_run_start(self, runner, policy):
        # Fail-fast: this callback is meaningless without a GS policy and the overlay wrapper.
        extractor = getattr(getattr(policy, "obs_encoder", None), "extractor", None)
        if not hasattr(extractor, "_latest_pool_indices"):
            raise LookupError(
                f"{type(self).__name__} requires a GSplat policy exposing "
                f"policy.obs_encoder.extractor._latest_pool_indices."
            )
        _find_wrapper(runner.env, AttentionOverlayWrapper)  # raises if absent
        self.num_unique_gaussians_over_all_episodes = []
        self.num_unique_gaussians_over_episode = []

    def before_episode_start_after_env_reset(self, runner, *, episode_idx, init_state, expert_trajectory):
        self.num_unique_gaussians_over_episode = []

    def after_action_predict(self, runner, *, obs_dict, action, action_dict, policy, env):
        pool_idx = policy.obs_encoder.extractor._latest_pool_indices
        if len(pool_idx.shape) > 1:
            pool_idx = pool_idx[-1]
        self.num_unique_gaussians_over_episode.append(len(torch.unique(pool_idx)))

        obs_gs_rgb = obs_dict['gs_rgb']
        if len(obs_gs_rgb.shape) > 1:
            obs_gs_rgb = obs_gs_rgb[-1]
        # Hand the attention off to the overlay wrapper, which recolors the scene at render.
        _find_wrapper(env, AttentionOverlayWrapper).set_attention(pool_idx, obs_gs_rgb)

    def at_episode_end(self, runner, *, episode_idx, is_success, prefix, env, expert_trajectory, info):
        self.num_unique_gaussians_over_all_episodes.append(np.array(self.num_unique_gaussians_over_episode))

    def at_run_end(self, runner, *, prefix, log_data):
        plot_dir = os.path.join(runner.output_dir, "eval_plots")
        os.makedirs(plot_dir, exist_ok=True)
        num_unique_gs_path = os.path.join(plot_dir, f"{prefix}_num_unique_gaussians_time.png")
        save_num_unique_gaussians_plot(self.num_unique_gaussians_over_all_episodes, save_path=num_unique_gs_path)
        cprint(f"Saved Num Unique Gaussians plot to {num_unique_gs_path}", "cyan")


# ---------------------------------------------------------------------------
# Trajectory plots (expert vs. actual vs. predicted)
# ---------------------------------------------------------------------------
class TrajectoryPlotCallback(EvalCallback):
    """Collects expert / actual / predicted trajectories and renders diagnostic plots.

    The plot set depends on `representation_space`:
      - abs_joint_pos:   phase-corridor + joint-position-over-time.
      - relative_ee_pose: TCP-state-over-time.
    """

    def __init__(self, representation_space: str):
        assert representation_space in ("abs_joint_pos", "relative_ee_pose"), \
            f"representation_space must be 'abs_joint_pos' or 'relative_ee_pose', got '{representation_space}'"
        self.representation_space = representation_space
        self.all_expert_trajectories: List = []
        self.all_policy_trajectories: List = []
        self.all_predicted_actions: List = []
        self.current_predicted_actions: List = []

    def at_run_start(self, runner, policy):
        self.all_expert_trajectories = []
        self.all_policy_trajectories = []
        self.all_predicted_actions = []
        self.current_predicted_actions = []

    def before_episode_start_after_env_reset(self, runner, *, episode_idx, init_state, expert_trajectory):
        self.current_predicted_actions = []

    def after_action_predict(self, runner, *, obs_dict, action, action_dict, policy, env):
        # What the policy wants the controller to reach (PD targets).
        if self.representation_space == "abs_joint_pos":
            self.current_predicted_actions.extend(action.cpu().numpy())

    def after_env_step(self, runner, *, obs, action, reward, done, info, policy, env):
        if self.representation_space == "relative_ee_pose":
            # RelativeEEControlWrapper stores the absolute TCP actions it actually issued.
            # (n_action_steps, 7) -> [x,y,z, euler_x,euler_y,euler_z, gripper]
            last_env_actions = _find_wrapper(env, RelativeEEControlWrapper).last_env_actions
            self.current_predicted_actions.extend(last_env_actions.cpu().numpy())

    def at_episode_end(self, runner, *, episode_idx, is_success, prefix, env, expert_trajectory, info):
        traj_agent_proprio = _find_wrapper(env, MultiStepWrapper).traj_agent_proprio
        self.all_expert_trajectories.append(expert_trajectory)
        self.all_policy_trajectories.append(torch.stack(list(traj_agent_proprio)).cpu().numpy())
        self.all_predicted_actions.append(np.array(self.current_predicted_actions))

    def at_run_end(self, runner, *, prefix, log_data):
        plot_dir = os.path.join(runner.output_dir, "eval_plots")
        os.makedirs(plot_dir, exist_ok=True)

        control_dt = 1.0 / _find_env_attr(runner.env, "control_freq")

        if self.representation_space == "abs_joint_pos":
            plot_path = os.path.join(plot_dir, f"{prefix}_phase_corridor.png")
            joint_pos_path = os.path.join(plot_dir, f"{prefix}_joint_pos_time.png")

            save_phase_corridor_plot(
                self.all_expert_trajectories,
                self.all_policy_trajectories,
                dt=control_dt,
                save_path=plot_path,
            )
            cprint(f"Saved Phase Corridor plot to {plot_path}", "cyan")

            save_joint_pos_over_time_plot(
                self.all_expert_trajectories,
                self.all_policy_trajectories,
                self.all_predicted_actions,
                save_path=joint_pos_path,
            )
            cprint(f"Saved Joint Position plot to {joint_pos_path}", "cyan")

        elif self.representation_space == "relative_ee_pose":
            tcp_state_path = os.path.join(plot_dir, f"{prefix}_tcp_state_time.png")
            save_tcp_state_over_time_plot(
                self.all_expert_trajectories,
                self.all_policy_trajectories,
                self.all_predicted_actions,
                save_path=tcp_state_path,
            )
            cprint(f"Saved TCP State plot to {tcp_state_path}", "cyan")


# ===========================================================================
# Plotting / rendering helpers
# ===========================================================================
def save_phase_corridor_plot(
    expert_trajectories: list,
    policy_trajectories: list,
    dt: float,
    save_path: str,
):
    """ Generates the Velocity vs Position Phase Plot for all joints."""

    if len(policy_trajectories) == 0:
        return None

    # Get num joints from the first trajectory
    if len(expert_trajectories) > 0 and expert_trajectories[0] is not None:
        num_joints = expert_trajectories[0].shape[1]
    else:
        num_joints = policy_trajectories[0].shape[1]

    fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(10, 3 * num_joints))
    if num_joints == 1: axes = [axes]
    fig.suptitle("Phase Corridor: Policy vs. Ground Truth", fontsize=16, fontweight='bold')

    for j in range(num_joints):
        ax = axes[j]

        # Plot all Policy Rollouts (Thin, translucent red)
        for i, p_q in enumerate(policy_trajectories):
            p_dq = np.gradient(p_q, dt, axis=0)
            p_dq[0] = 0.0   # np.gradient is inaccurate for t=0. Maniskill resets agent to zero velocity!
            label = 'Policy Rollout' if i == 0 else "_nolegend_"
            ax.plot(p_q[:, j], p_dq[:, j], color='red', alpha=0.3, linewidth=1.0, label=label)

        # Plot all Expert Ground Truths (Thick, black, dashed)
        for i, e_q in enumerate(expert_trajectories):
            if e_q is not None:
                e_dq = np.gradient(e_q, dt, axis=0)
                e_dq[0] = 0.0   # np.gradient is inaccurate for t=0. Maniskill resets agent to zero velocity!
                label = 'Motion Planner (Expert)' if i == 0 else "_nolegend_"
                ax.plot(e_q[:, j], e_dq[:, j], color='black', linewidth=2, label=label)
                # Mark start and end targets
                ax.scatter(e_q[0, j], e_dq[0, j], c='green', marker='o', s=100, zorder=5, label='Start' if i == 0 else "_nolegend_")
                ax.scatter(e_q[-1, j], e_dq[-1, j], c='black', marker='*', s=150, zorder=5, label='End Target' if i == 0 else "_nolegend_")

        ax.set_title(f"Joint {j}")
        ax.set_ylabel("Velocity [rad/s]")
        ax.grid(True, alpha=0.3)
        if j == 0: ax.legend(loc='upper right')
        if j == num_joints - 1: ax.set_xlabel("Position [rad]")

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def save_joint_pos_over_time_plot(
    expert_trajectories: list,
    policy_trajectories: list,
    predicted_actions: list = None,
    save_path: str = "joint_pos_time.png",
):
    """ Generates the Position vs Timestep Plot for all joints.

    Shows three lines per joint:
    - Ground Truth (expert): black solid line
    - Predicted Actions (policy targets): blue dashed — what the policy wants
    - Actual qpos (after PD control): red solid — what actually happened

    The gap between predicted and actual reveals PD execution error.
    The gap between ground truth and predicted reveals policy prediction error.
    """
    if len(policy_trajectories) == 0:
        return None

    # Get num joints from the first trajectory
    if len(expert_trajectories) > 0 and expert_trajectories[0] is not None:
        num_joints = expert_trajectories[0].shape[1]
    else:
        num_joints = policy_trajectories[0].shape[1]

    fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(12, 3 * num_joints))
    if num_joints == 1: axes = [axes]
    fig.suptitle("Joint Position over Time: Predicted vs. Actual vs. Ground Truth", fontsize=16, fontweight='bold')

    for j in range(num_joints):
        ax = axes[j]

        # Plot all Actual qpos
        for i, p_q in enumerate(policy_trajectories):
            time_axis = np.arange(len(p_q))
            label = 'Actual qpos (after PD)' if i == 0 else "_nolegend_"
            ax.plot(time_axis, p_q[:, j], color='red', alpha=0.3, linewidth=1.0, label=label)

        # Plot all Expert Ground Truths (Thick, black)
        for i, e_q in enumerate(expert_trajectories):
            if e_q is not None:
                time_axis = np.arange(len(e_q))
                label = 'Motion Planner (Expert)' if i == 0 else "_nolegend_"
                ax.plot(time_axis, e_q[:, j], color='black', linewidth=2, label=label)

        # Plot all Predicted Actions
        for i, p_a in enumerate(predicted_actions):
            if p_a is not None and j < p_a.shape[1]:
                time_axis = np.arange(1, len(p_a) + 1)
                label = 'Predicted Action (PD target)' if i == 0 else "_nolegend_"
                ax.plot(time_axis, p_a[:, j], color='dodgerblue', alpha=0.4, linewidth=1.0, linestyle='--', label=label)

        ax.set_title(f"Joint {j}")
        ax.set_ylabel("Position [rad]")
        ax.grid(True, alpha=0.3)
        if j == 0: ax.legend(loc='upper right')
        if j == num_joints - 1: ax.set_xlabel("Timestep")

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def _quat_wxyz_to_euler_xyz_deg(quat_wxyz):
    """Convert quaternion [qw,qx,qy,qz] array to unwrapped Euler XYZ angles in degrees.

    Applies np.unwrap along the time axis to remove ±180° wrapping discontinuities
    that are inherent to the Euler angle representation.

    Args:
        quat_wxyz: np.ndarray of shape (T, 4) with [qw, qx, qy, qz]
    Returns:
        euler_deg: np.ndarray of shape (T, 3) with [roll, pitch, yaw] in degrees (unwrapped)
    """

    original_shape = quat_wxyz.shape[:-1]
    quat_flat = quat_wxyz.reshape(-1, 4)
    # scipy expects [qx, qy, qz, qw], our data is [qw, qx, qy, qz]
    quat_xyzw = quat_flat[:, [1, 2, 3, 0]]
    euler_rad = Rotation.from_quat(quat_xyzw).as_euler('XYZ')
    # Unwrap each angle column to remove ±π discontinuities before converting to degrees
    euler_rad = np.unwrap(euler_rad, axis=0)
    euler_deg = np.degrees(euler_rad)
    return euler_deg.reshape(original_shape + (3,))


def save_tcp_state_over_time_plot(
    expert_trajectories: list,
    policy_trajectories: list,
    predicted_actions: list = None,
    save_path: str = "tcp_state_time.png",
):
    """ Generates Position, Orientation (Euler), and Gripper State over Timestep plots.

    Data formats:
    - expert_trajectories: list of (T, 9) arrays [x,y,z, qw,qx,qy,qz, grip1, grip2]
    - policy_trajectories: list of (T, 9) arrays [x,y,z, qw,qx,qy,qz, grip1, grip2]
    - predicted_actions: list of (T, 8) arrays [x,y,z, qw,qx,qy,qz, gripper]

    Subplots (8 total):
    - Position: x, y, z (meters)
    - Orientation: roll, pitch, yaw (degrees, converted from quaternion for expert/actual)
    - Gripper: grip1, grip2 (only 1D gripper action for predicted)
    """
    if len(policy_trajectories) == 0:
        return None

    pos_labels = ['x [m]', 'y [m]', 'z [m]']
    rot_labels = ['Roll [deg] (unwrapped)', 'Pitch [deg] (unwrapped)', 'Yaw [deg] (unwrapped)']
    grip_labels = ['Gripper 1', 'Gripper 2']
    num_subplots = 8  # 3 pos + 3 rot + 2 gripper

    fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(12, 3 * num_subplots))
    fig.suptitle("TCP State over Time: Predicted vs. Actual vs. Ground Truth", fontsize=16, fontweight='bold')

    # --- Position subplots (indices 0, 1, 2) ---
    for dim in range(3):
        ax = axes[dim]

        for i, p_traj in enumerate(policy_trajectories):
            time_axis = np.arange(len(p_traj))
            label = 'Actual TCP Pose' if i == 0 else "_nolegend_"
            ax.plot(time_axis, p_traj[:, dim], color='red', alpha=0.3, linewidth=1.0, label=label)

        for i, e_traj in enumerate(expert_trajectories):
            if e_traj is not None:
                time_axis = np.arange(len(e_traj))
                label = 'Motion Planner (Expert)' if i == 0 else "_nolegend_"
                ax.plot(time_axis, e_traj[:, dim], color='black', linewidth=2, label=label)

        if predicted_actions is not None:
            for i, p_a in enumerate(predicted_actions):
                if p_a is not None:
                    time_axis = np.arange(1, len(p_a) + 1)
                    label = 'Predicted Action (EE target)' if i == 0 else "_nolegend_"
                    ax.plot(time_axis, p_a[:, dim], color='dodgerblue', alpha=0.4, linewidth=1.0, linestyle='--', label=label)

        ax.set_title(f"Position: {pos_labels[dim]}")
        ax.set_ylabel(pos_labels[dim])
        ax.grid(True, alpha=0.3)
        if dim == 0: ax.legend(loc='upper right')

    # --- Orientation subplots (indices 3, 4, 5) ---
    for dim in range(3):
        ax = axes[3 + dim]

        for i, p_traj in enumerate(policy_trajectories):
            # policy_trajectories has quaternion [qw,qx,qy,qz] at indices 3:7
            euler_deg = _quat_wxyz_to_euler_xyz_deg(p_traj[:, 3:7])
            time_axis = np.arange(len(p_traj))
            label = 'Actual TCP Orientation' if i == 0 else "_nolegend_"
            ax.plot(time_axis, euler_deg[:, dim], color='red', alpha=0.3, linewidth=1.0, label=label)

        for i, e_traj in enumerate(expert_trajectories):
            if e_traj is not None:
                euler_deg = _quat_wxyz_to_euler_xyz_deg(e_traj[:, 3:7])
                time_axis = np.arange(len(e_traj))
                label = 'Motion Planner (Expert)' if i == 0 else "_nolegend_"
                ax.plot(time_axis, euler_deg[:, dim], color='black', linewidth=2, label=label)

        if predicted_actions is not None:
            for i, p_a in enumerate(predicted_actions):
                if p_a is not None:
                    euler_deg = _quat_wxyz_to_euler_xyz_deg(p_a[:, 3:7])
                    time_axis = np.arange(1, len(p_a) + 1)
                    label = 'Predicted Action (EE target)' if i == 0 else "_nolegend_"
                    ax.plot(time_axis, euler_deg[:, dim], color='dodgerblue', alpha=0.4, linewidth=1.0, linestyle='--', label=label)

        ax.set_title(f"Orientation: {rot_labels[dim]}")
        ax.set_ylabel(rot_labels[dim])
        ax.grid(True, alpha=0.3)
        if dim == 0: ax.legend(loc='upper right')

    # --- Gripper subplots (indices 6, 7) ---
    for dim in range(2):
        ax = axes[6 + dim]

        for i, p_traj in enumerate(policy_trajectories):
            # Gripper state at indices 7:9
            time_axis = np.arange(len(p_traj))
            label = 'Actual Gripper State' if i == 0 else "_nolegend_"
            ax.plot(time_axis, p_traj[:, 7 + dim], color='red', alpha=0.3, linewidth=1.0, label=label)

        for i, e_traj in enumerate(expert_trajectories):
            if e_traj is not None:
                time_axis = np.arange(len(e_traj))
                label = 'Motion Planner (Expert)' if i == 0 else "_nolegend_"
                ax.plot(time_axis, e_traj[:, 7 + dim], color='black', linewidth=2, label=label)

        if predicted_actions is not None:
            # Predicted actions only have 1D gripper (index 7), but we plot it on both gripper subplots
            for i, p_a in enumerate(predicted_actions):
                if p_a is not None and p_a.shape[1] > 7:
                    time_axis = np.arange(1, len(p_a) + 1)
                    label = 'Predicted Action (gripper target)' if i == 0 else "_nolegend_"
                    ax.plot(time_axis, p_a[:, 7], color='dodgerblue', alpha=0.4, linewidth=1.0, linestyle='--', label=label)

        ax.set_title(f"Gripper: {grip_labels[dim]}")
        ax.set_ylabel(grip_labels[dim])
        ax.grid(True, alpha=0.3)
        if dim == 0: ax.legend(loc='upper right')
        if dim == 1: ax.set_xlabel("Timestep")

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def save_num_unique_gaussians_plot(
    all_num_unique_gaussians: list,
    save_path: str,
):
    """ Plots median, 25%, and 75% quantile of num unique Gaussians over time. """
    if len(all_num_unique_gaussians) == 0:
        return None

    # Find max length
    max_len = max(len(traj) for traj in all_num_unique_gaussians)

    # Pad with NaN
    padded_data = np.full((len(all_num_unique_gaussians), max_len), np.nan)
    for i, traj in enumerate(all_num_unique_gaussians):
        padded_data[i, :len(traj)] = traj

    # Calculate quantiles ignoring NaNs
    q25 = np.nanpercentile(padded_data, 25, axis=0)
    q50 = np.nanpercentile(padded_data, 50, axis=0)  # median
    q75 = np.nanpercentile(padded_data, 75, axis=0)

    time_axis = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_axis, q50, color='blue', linewidth=2, label='Median Unique Gaussians')
    ax.fill_between(time_axis, q25, q75, color='blue', alpha=0.2, label='25%-75% Quantile')

    ax.set_title("Unique Selected Gaussians over Time", fontsize=14, fontweight='bold')
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of Unique Gaussians")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def add_legend_to_video(video_frames, colormap='cool'):
    """
    Appends a legend (colorbar) to the bottom of the video frames.
    video_frames: (T, H, W, C) numpy array of uint8
    """

    T, H, W, C = video_frames.shape
    dpi = 100
    fig_height = 1.0  # inches
    fig = Figure(figsize=(W / dpi, fig_height), dpi=dpi)
    canvas = FigureCanvasAgg(fig)

    # Create axes for colorbar
    ax = fig.add_axes([0.1, 0.4, 0.8, 0.2])  # left, bottom, width, height

    cmap = plt.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=0, vmax=1)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax, orientation='horizontal')
    cb.set_label('Policy Attention Frequency (Normalized)', fontsize=8)
    cb.ax.tick_params(labelsize=8)

    # Add descriptive text
    fig.text(0.5, 0.8, "Magenta: High Attention | Cyan: Low Attention | Yellow: Ignored",
             ha='center', va='center', fontsize=9, fontweight='bold')

    # Render to numpy array
    canvas.draw()
    colorbar_img = np.asarray(canvas.buffer_rgba())[..., :3]  # Get RGB channels

    # Ensure width matches exactly
    ch, cw, _ = colorbar_img.shape
    if cw != W:
        colorbar_img = cv2.resize(colorbar_img, (W, ch))

    # Tile across all frames
    colorbar_video = np.tile(colorbar_img[None, ...], (T, 1, 1, 1))

    # Concatenate vertically
    return np.concatenate([video_frames, colorbar_video], axis=1)
