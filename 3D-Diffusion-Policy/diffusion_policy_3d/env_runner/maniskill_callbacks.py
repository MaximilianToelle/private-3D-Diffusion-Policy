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
import wandb

from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.env.maniskill.relative_ee_control_wrapper import RelativeEEControlWrapper
from diffusion_policy_3d.env.maniskill.attention_overlay_wrapper import AttentionOverlayWrapper
from diffusion_policy_3d.env.maniskill.actor_pose_recording_wrapper import ActorPoseRecordingWrapper


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
    The wrapper captures one frame per control step, so playing the video at the control frequency
    shows it in real time. The Gaussian-attention colorbar legend is appended below the frames
    whenever a `GaussianAttentionPlotCallback` runs, the only case the frames are colored by
    attention. Episode 0 of every run also goes to wandb as `video`, at half the resolution.

    Args:
        colormap: matplotlib colormap name for the legend.
    """

    def __init__(self, colormap: str = 'cool'):
        self.colormap = colormap

    def at_episode_end(self, runner, *, episode_idx, is_success, prefix, env, expert_trajectory, info):
        video = _find_wrapper(env, SimpleVideoRecordingWrapper).get_video()  # (T, C, H, W)
        if episode_idx == 0:
            self.wandb_video = wandb.Video(video[:, :, ::2, ::2], fps=_find_env_attr(runner.env, "control_freq"), format="mp4")

        video_dir = os.path.join(runner.output_dir, "eval_videos")
        os.makedirs(video_dir, exist_ok=True)
        video_to_save = video.transpose(0, 2, 3, 1)  # (T, H, W, C)
        if any(isinstance(callback, GaussianAttentionPlotCallback) for callback in runner.callbacks):
            video_to_save = add_legend_to_video(video_to_save, colormap=self.colormap)

        suffix = "success" if is_success else "failure"
        out_path = os.path.join(video_dir, f"{prefix}_ep_{episode_idx}_{suffix}.mp4")
        imageio.mimsave(out_path, video_to_save, fps=_find_env_attr(runner.env, "control_freq"), macro_block_size=1)
        cprint(f"Saved evaluation video to {out_path}", "cyan")

    def at_run_end(self, runner, *, prefix, log_data):
        log_data['video'] = self.wandb_video


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
        self.all_expert_trajectories.append(expert_trajectory['agent_proprio'])
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


# ---------------------------------------------------------------------------
# Grasp and insertion deviation from the demonstration
# ---------------------------------------------------------------------------
def _pose_matrix(pose7):
    """[x, y, z, qw, qx, qy, qz] -> (4, 4)."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(pose7[3:7])[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = pose7[:3]
    return matrix


def _pose_change(pose_from, pose_to):
    """Translation [mm] and rotation angle [deg] between two (4, 4) poses."""
    delta = np.linalg.inv(pose_from) @ pose_to
    return 1000 * np.linalg.norm(delta[:3, 3]), np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude())


def _yaw_deg(pose7):
    """Yaw of a pose rotated about z only, in [-180, 180)."""
    return (np.degrees(2 * np.arctan2(pose7[6], pose7[3])) + 180) % 360 - 180


def _demonstration_grasp_and_release(gripper_action):
    """Steps of the demonstration's grasp (first gripper action below 0) and release (first one
    above 0 after the grasp, None if the gripper never opens again). The demonstrations ramp the
    action between 1, open, and -1, closed."""
    grasp = int(np.flatnonzero(gripper_action < 0)[0])
    opening = np.flatnonzero(gripper_action[grasp:] > 0)
    return grasp, grasp + int(opening[0]) if len(opening) > 0 else None


class GripperEvents:
    """The policy's grasp (first executed gripper action below 0) and release (first one above
    0 after the grasp), as in _demonstration_grasp_and_release. Callbacks read the env at the end
    of an action chunk, so each event comes with the number of steps the chunk executed from it
    on."""

    def reset(self):
        self.closed = self.released = False

    def chunk_predicted(self, action, env):
        self.chunk_gripper_actions = action[:, -1].cpu().numpy()
        self.steps_before_chunk = len(_find_wrapper(env, MultiStepWrapper).traj_agent_proprio) - 1

    def chunk_executed(self, env):
        """The steps executed from the grasp and from the release on, each None unless it
        happened in this chunk."""
        executed = len(_find_wrapper(env, MultiStepWrapper).traj_agent_proprio) - 1 - self.steps_before_chunk
        actions = self.chunk_gripper_actions[:executed]
        steps_since_grasp = steps_since_release = None
        if not self.closed:
            closing = np.flatnonzero(actions < 0)
            if len(closing) == 0:
                return None, None
            self.closed, steps_since_grasp = True, executed - closing[0]
            actions = actions[closing[0]:]
        if not self.released:
            opening = np.flatnonzero(actions > 0)
            if len(opening) > 0:
                self.released, steps_since_release = True, len(actions) - opening[0]
        return steps_since_grasp, steps_since_release


class GraspDeviationCallback(EvalCallback):
    """How precisely the policy grasps the piece, next to the same for the demonstration of the
    same start state (every rollout starts from a dataset episode's start state, so that
    episode's demonstration is the oracle's grasp for exactly this start). The close command is
    the first gripper action below 0 (GripperEvents).
      grasp pose error     tcp pose of the policy vs the demonstration, each at the step its
                           close command is issued, before the fingers move; the policy's by
                           forward kinematics of the joint positions MultiStepWrapper records
                           every step, since callbacks read the env at action-chunk ends only
      grasp piece displacement
                           how far the gripper moves the piece while grasping it: translation and
                           rotation angle of the piece from its start pose to the step at which the
                           fingers have stopped closing, or to the last step before the piece rises
                           if that comes first (the demonstrations stop the fingers 8 steps after
                           the close command and start the lift one step later, moving the piece
                           sideways with the hand one step before it rises); from the poses
                           ActorPoseRecordingWrapper records every step; the same for the
                           demonstration
    Logs median / p5 / p95 / max of the policy's metrics and how many rollouts closed the
    gripper, and plots every rollout over the piece's start yaw, with the start yaws of the
    training episodes as ticks.

    Requires the joint positions as proprio (abs_joint_pos), ActorPoseRecordingWrapper and a
    dataset whose expert trajectory holds tcp_pose, gripper_action and actor_poses
    (WristCamPCDMemmapManiskillDataset).
    """

    PANELS = [[('grasp_position_error_mm', 'grasp pose error: position [mm]'),
               ('grasp_rotation_error_deg', 'grasp pose error: rotation [deg]')],
              [('grasp_piece_displacement_mm', 'grasp piece displacement: translation [mm]'),
               ('grasp_piece_displacement_deg', 'grasp piece displacement: rotation [deg]')]]
    POLICY_COLOR, DEMONSTRATION_COLOR, TICK_COLOR = '#2a78d6', '#eb6834', '#9a9a96'
    FINGERS_STOPPED = 1e-5  # [m] change of the finger opening in one step below which the fingers have stopped closing
    LIFT_HEIGHT = 1e-4      # [m] rise above the start height from which the piece counts as lifted; closing raises it <= 0.04 mm

    def __init__(self, actor_name: str):
        self.actor_name = actor_name
        self.actor_key = f'actor_pose_{actor_name}'
        self.gripper = GripperEvents()
        self.rollouts: List[dict] = []

    def at_run_start(self, runner, policy):
        assert runner.representation_space == "abs_joint_pos", "the grasp pose error needs the joint positions as proprio"
        self.rollouts = []

    def before_episode_start_after_env_reset(self, runner, *, episode_idx, init_state, expert_trajectory):
        self.demonstration = expert_trajectory
        self.start_yaw = _yaw_deg(init_state['actor_poses'][self.actor_key])
        self.gripper.reset()
        self.close_step = self.tcp_at_close = None

    def after_action_predict(self, runner, *, obs_dict, action, action_dict, policy, env):
        self.gripper.chunk_predicted(action, env)

    @staticmethod
    def _tcp_pose(env, joint_positions):
        """(4, 4) world pose of the tcp at these joint positions."""
        robot = _find_env_attr(env, "agent").robot
        model = robot.create_pinocchio_model()
        model.compute_forward_kinematics(joint_positions.cpu().numpy())
        tcp = model.get_link_pose([link.name for link in robot.get_links()].index(_find_env_attr(env, "agent").tcp.name))
        return _pose_matrix(robot.pose.raw_pose[0].cpu().numpy()) @ _pose_matrix(np.concatenate([tcp.p, tcp.q]))

    @classmethod
    def _piece_displacement(cls, piece_poses, joint_positions, close_step):
        """Translation [mm] and rotation [deg] of the piece from its first pose to the step at
        which the fingers have stopped closing after close_step, or to the last step before it
        rises if that comes first. piece_poses (steps, 7) and joint_positions (steps, 9), [i]
        before the i-th action."""
        last = len(piece_poses) - 1
        finger_opening = joint_positions[close_step:, 7] + joint_positions[close_step:, 8]
        stopped = np.flatnonzero(np.abs(np.diff(finger_opening)) < cls.FINGERS_STOPPED)
        risen = np.flatnonzero(piece_poses[:, 2] - piece_poses[0, 2] > cls.LIFT_HEIGHT)
        end = min(close_step + 1 + stopped[0] if len(stopped) > 0 else last, risen[0] - 1 if len(risen) > 0 else last)
        return _pose_change(_pose_matrix(piece_poses[0]), _pose_matrix(piece_poses[end]))

    def after_env_step(self, runner, *, obs, action, reward, done, info, policy, env):
        steps_since_close, _ = self.gripper.chunk_executed(env)
        if steps_since_close is not None:
            joint_positions = _find_wrapper(env, MultiStepWrapper).traj_agent_proprio  # [i]: before the i-th action
            self.close_step = len(joint_positions) - 1 - steps_since_close
            self.tcp_at_close = self._tcp_pose(env, joint_positions[self.close_step])

    def at_episode_end(self, runner, *, episode_idx, is_success, prefix, env, expert_trajectory, info):
        rollout = dict(start_yaw=self.start_yaw, is_success=is_success, closed=self.close_step is not None)
        if self.close_step is not None:
            demonstration_close, _ = _demonstration_grasp_and_release(self.demonstration['gripper_action'])
            rollout['grasp_position_error_mm'], rollout['grasp_rotation_error_deg'] = _pose_change(
                _pose_matrix(self.demonstration['tcp_pose'][demonstration_close]), self.tcp_at_close)
            joint_positions = np.stack([qpos.cpu().numpy() for qpos in _find_wrapper(env, MultiStepWrapper).traj_agent_proprio])
            rollout['grasp_piece_displacement_mm'], rollout['grasp_piece_displacement_deg'] = self._piece_displacement(
                _find_wrapper(env, ActorPoseRecordingWrapper).get_actor_poses(self.actor_name), joint_positions, self.close_step)
            rollout['demonstration_grasp_piece_displacement_mm'], rollout['demonstration_grasp_piece_displacement_deg'] = \
                self._piece_displacement(self.demonstration['actor_poses'][self.actor_key],
                                         self.demonstration['agent_proprio'], demonstration_close)
        self.rollouts.append(rollout)

    def at_run_end(self, runner, *, prefix, log_data):
        closed = [rollout for rollout in self.rollouts if rollout['closed']]
        log_data['grasp_num_closed'] = len(closed)
        log_data['grasp_num_never_closed'] = len(self.rollouts) - len(closed)
        _log_spread(log_data, closed, [metric for row in self.PANELS for metric, _ in row])
        _save_deviation_plot_of_run(runner, self, closed, prefix, "grasp_deviation",
                                    title=f"{prefix}: {len(closed)} of {len(self.rollouts)} rollouts closed the gripper")


class InsertionDeviationCallback(EvalCallback):
    """How far the piece is from its hole above the hole and at the policy's release, next to
    the same for the demonstration of the same start state (InsertFmbObjectFr3Env).

      above hole  the last action-chunk end at which the lifted piece's bottom is still above
                  the board's top face, before it comes down to it (the env is read at chunk ends
                  only, so the piece is up to one chunk's motion above the face); for the
                  demonstration, its last step before the piece comes down
      release     the policy's release (GripperEvents), read at the end of its action chunk, k
                  steps after it, the demonstration k steps after its own release
    At each moment: lateral offset of the piece frame's origin from the hole's in xy [mm], yaw
    error modulo the shape's symmetry period [deg] and tilt [deg]; at the release also the height
    above the seated pose [mm] (0: fully inserted, the board's height: on its top face). The
    hole is the env's hole_pose, the pose of the seated piece; the board does not move during an
    episode, so it is the demonstration's too.
    Logs median / p5 / p95 / max of the policy's metrics and how many rollouts reached each
    moment, and plots every rollout over the piece's start yaw like GraspDeviationCallback.
    """

    PANELS = [[('above_hole_lateral_offset_mm', 'above hole: lateral offset [mm]'),
               ('above_hole_yaw_error_deg', 'above hole: yaw error [deg]'),
               ('above_hole_tilt_deg', 'above hole: tilt [deg]')],
              [('release_lateral_offset_mm', 'release: lateral offset [mm]'),
               ('release_yaw_error_deg', 'release: yaw error [deg]'),
               ('release_tilt_deg', 'release: tilt [deg]'),
               ('release_height_above_seated_mm', 'release: height above seated [mm]')]]
    POLICY_COLOR, DEMONSTRATION_COLOR, TICK_COLOR = GraspDeviationCallback.POLICY_COLOR, GraspDeviationCallback.DEMONSTRATION_COLOR, GraspDeviationCallback.TICK_COLOR
    TOP_FACE_TOLERANCE = 0.001  # [m] above the board's top face that still counts as on it

    def __init__(self, actor_name: str):
        self.actor_name = actor_name
        self.actor_key = f'actor_pose_{actor_name}'
        self.gripper = GripperEvents()
        self.rollouts: List[dict] = []

    def at_run_start(self, runner, policy):
        self.rollouts = []

    def before_episode_start_after_env_reset(self, runner, *, episode_idx, init_state, expert_trajectory):
        self.demonstration = expert_trajectory
        self.start_yaw = _yaw_deg(init_state['actor_poses'][self.actor_key])
        self.gripper.reset()
        self.hole = self.last_above_face = self.above_hole = self.release = None

    def after_action_predict(self, runner, *, obs_dict, action, action_dict, policy, env):
        self.gripper.chunk_predicted(action, env)
        if self.hole is None:
            self.hole = _pose_matrix(_find_env_attr(env, "hole_pose").raw_pose[0].cpu().numpy())
            self.symmetry_period_deg = _find_env_attr(env, "SYMMETRY_PERIOD_DEG")[_find_env_attr(env, "piece").shape]
            self.board_height = _find_env_attr(env, "board").HEIGHT

    def _above_face(self, pieces):
        return pieces[..., 2, 3] - self.hole[2, 3] > self.board_height + self.TOP_FACE_TOLERANCE

    def _errors(self, piece, moment):
        """Every metric of the piece's pose relative to the hole, as <moment>_<metric>."""
        relative = self.hole[:3, :3].T @ piece[:3, :3]
        yaw = np.degrees(np.arctan2(relative[1, 0], relative[0, 0]))
        period = self.symmetry_period_deg
        return {f'{moment}_lateral_offset_mm': 1000 * np.linalg.norm(piece[:2, 3] - self.hole[:2, 3]),
                f'{moment}_yaw_error_deg': abs((yaw + period / 2) % period - period / 2),
                f'{moment}_tilt_deg': np.degrees(np.arccos(np.clip(relative[2, 2], -1, 1))),
                f'{moment}_height_above_seated_mm': 1000 * (piece[2, 3] - self.hole[2, 3])}

    def after_env_step(self, runner, *, obs, action, reward, done, info, policy, env):
        _, steps_since_release = self.gripper.chunk_executed(env)
        piece = _pose_matrix(_find_env_attr(env, "scene").actors[self.actor_name].pose.raw_pose[0].cpu().numpy())
        if steps_since_release is not None:
            self.release = piece, steps_since_release
        if self.above_hole is None:
            if self._above_face(piece):
                self.last_above_face = piece
            elif self.last_above_face is not None:
                self.above_hole = self.last_above_face

    def _demonstration_moments(self, steps_since_release):
        pieces = np.stack([_pose_matrix(pose) for pose in self.demonstration['actor_poses'][self.actor_key]])
        above_face = self._above_face(pieces)
        lifted = int(np.flatnonzero(above_face)[0])
        above_hole = lifted + int(np.flatnonzero(~above_face[lifted:])[0]) - 1
        _, release = _demonstration_grasp_and_release(self.demonstration['gripper_action'])
        return pieces[above_hole], pieces[min(release + steps_since_release, len(pieces) - 1)]

    def at_episode_end(self, runner, *, episode_idx, is_success, prefix, env, expert_trajectory, info):
        rollout = dict(start_yaw=self.start_yaw, is_success=is_success)
        demonstration_above_hole, demonstration_release = self._demonstration_moments(0 if self.release is None else self.release[1])
        policy_release = None if self.release is None else self.release[0]
        for moment, policy_piece, demonstration_piece in (('above_hole', self.above_hole, demonstration_above_hole),
                                                          ('release', policy_release, demonstration_release)):
            if policy_piece is not None:
                rollout.update(self._errors(policy_piece, moment))
            rollout.update({f'demonstration_{key}': value for key, value in self._errors(demonstration_piece, moment).items()})
        self.rollouts.append(rollout)

    def at_run_end(self, runner, *, prefix, log_data):
        log_data['insertion_num_above_hole'] = sum('above_hole_lateral_offset_mm' in rollout for rollout in self.rollouts)
        log_data['insertion_num_released'] = sum('release_lateral_offset_mm' in rollout for rollout in self.rollouts)
        _log_spread(log_data, self.rollouts, [metric for row in self.PANELS for metric, _ in row])
        _save_deviation_plot_of_run(runner, self, self.rollouts, prefix, "insertion_deviation",
                                    title=(f"{prefix}: {log_data['insertion_num_above_hole']} of {len(self.rollouts)} rollouts came down "
                                           f"from above the board, {log_data['insertion_num_released']} released the piece"))


def _log_spread(log_data, rollouts, metrics):
    """median / p5 / p95 / max of each metric over the rollouts that hold it."""
    for metric in metrics:
        values = np.array([rollout[metric] for rollout in rollouts if metric in rollout])
        if len(values) > 0:
            log_data[f'{metric}_median'] = np.median(values)
            log_data[f'{metric}_p5'], log_data[f'{metric}_p95'] = np.percentile(values, [5, 95])
            log_data[f'{metric}_max'] = values.max()


def _save_deviation_plot_of_run(runner, callback, rollouts, prefix, name, title):
    dataset = runner.rollout_dataset
    training_start_yaws = dataset.episode_start_yaws_deg(callback.actor_key)[dataset.training_episode_mask]
    plot_dir = os.path.join(runner.output_dir, "eval_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{prefix}_{name}.png")
    save_deviation_plot(rollouts, callback.PANELS, training_start_yaws, callback.POLICY_COLOR, callback.DEMONSTRATION_COLOR,
                        callback.TICK_COLOR, title=title, save_path=plot_path)
    cprint(f"Saved {name.replace('_', ' ')} plot to {plot_path}", "cyan")


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


def save_deviation_plot(rollouts, panel_rows, training_start_yaws, policy_color, demonstration_color, tick_color, title, save_path):
    """One panel per metric over the actor's start yaw, a row of panels per row of panel_rows:
    the policy's rollouts that hold the metric (x = failed), the demonstration's value as the
    reference where the rollouts hold one, the training episodes' start yaws as ticks."""
    num_columns = max(len(panels) for panels in panel_rows)
    fig, axes = plt.subplots(len(panel_rows), num_columns, figsize=(5.5 * num_columns, 3.5 * len(panel_rows)), sharex=True, squeeze=False)
    for row, panels in zip(axes, panel_rows):
        for ax in row[len(panels):]:
            ax.set_visible(False)
        for ax, (metric, label) in zip(row, panels):
            for is_success, marker in ((True, 'o'), (False, 'x')):
                subset = [rollout for rollout in rollouts if metric in rollout and rollout['is_success'] == is_success]
                ax.scatter([rollout['start_yaw'] for rollout in subset], [rollout[metric] for rollout in subset],
                           marker=marker, s=40, color=policy_color, linewidths=1.5,
                           label=f"policy, {'success' if is_success else 'failure'}")
            reference = [rollout for rollout in rollouts if f'demonstration_{metric}' in rollout]
            if reference:
                ax.scatter([rollout['start_yaw'] for rollout in reference], [rollout[f'demonstration_{metric}'] for rollout in reference],
                           marker='s', s=40, facecolors='none', edgecolors=demonstration_color, linewidths=1.5,
                           label='demonstration (reference)')
            ax.plot(training_start_yaws, np.zeros_like(training_start_yaws), '|', markersize=10, color=tick_color,
                    transform=ax.get_xaxis_transform(), label='training start yaws')
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            ax.grid(True, color='#e6e6e3', linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('start yaw [deg]')
    handles, labels = axes[-1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
