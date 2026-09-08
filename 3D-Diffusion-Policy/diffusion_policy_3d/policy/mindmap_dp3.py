"""mindmap baseline policy: thin BasePolicy adapter around the vendored
DiffuserActor (3D Diffuser Actor on nvblox feature clouds), so it trains and
evaluates inside the DP3 harness.

Batch contract (from WristCamNvbloxManiskillDataset):
  obs.vertices            (B, N, 3)   raw world frame
  obs.vertex_features     (B, N, C)
  obs.vertices_valid_mask (B, N)
  obs.gripper_history     (B, nhist, 8)   [pos3, quat_wxyz4, closedness1]
  action                  (B, npred, 8)   next keypose(s), same layout

Normalization happens INSIDE DiffuserActor via fixed workspace bounds; the
harness LinearNormalizer is identity and stored only for interface compat.
"""

from typing import Dict, List

import torch

import math

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.model.mindmap.data_types import DataType
from diffusion_policy_3d.model.mindmap.diffuser_actor import DiffuserActor
from diffusion_policy_3d.model.mindmap.embodiments.arm.gripper import GRIPPER_OPEN_THRESHOLD
from diffusion_policy_3d.model.mindmap.embodiments.delay_based_estimator import (
    DelayBasedGripperStateEstimator,
)
from diffusion_policy_3d.model.mindmap.feature_types import FeatureExtractorType
from diffusion_policy_3d.model.mindmap.geometry.pytorch3d_transforms import (
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_axis_angle,
)
from diffusion_policy_3d.model.mindmap.loss import LossWeights
from diffusion_policy_3d.policy.base_policy import BasePolicy


class MindmapDP3(BasePolicy):
    def __init__(
        self,
        feature_type: str = "radio_v25_b",
        embedding_dim: int = 120,
        fps_subsampling_factor: int = 5,
        use_fps: int = 1,
        rotation_parametrization: str = "6D_from_query",
        quaternion_format: str = "wxyz",
        diffusion_timesteps: int = 100,
        num_history: int = 3,
        prediction_horizon: int = 1,
        pos_loss_weight: float = 30.0,
        rot_loss_weight: float = 10.0,
        gripper_loss_weight: float = 1.0,
        encoder_dropout: float = 0.0,
        diffusion_dropout: float = 0.0,
        predictor_dropout: float = 0.0,
        gripper_open_action: float = 1.0,
        gripper_close_action: float = -1.0,
        max_num_steps_to_goal: int = 40,
        goal_reached_threshold_m: float = 0.001,
        goal_reached_threshold_deg: float = 1.0,
        goal_reached_threshold_gripper_diff: float = 0.2,
        gripper_command_delay_steps: int = 10,
    ):
        super().__init__()
        self.num_history = num_history
        self.prediction_horizon = prediction_horizon
        # Closed-loop execution replicating mindmap's CLOSED_LOOP_WAIT: re-run
        # inference when the current goal is reached (pos/rot/gripper thresholds
        # from their arm constants) or after max_num_steps_to_goal control steps;
        # otherwise keep commanding the same absolute goal. Runs with
        # n_action_steps=1 so this is evaluated every control step. Gripper
        # command range of pd_ee_pos_quat is [-1, 1] with +1 = open; the current
        # gripper state is estimated delay-based from past commands (their
        # ArmEmbodimentOnlineEstimator), not from jaw positions.
        self.gripper_open_action = gripper_open_action
        self.gripper_close_action = gripper_close_action
        self.max_num_steps_to_goal = max_num_steps_to_goal
        self.goal_reached_threshold_m = goal_reached_threshold_m
        self.goal_reached_threshold_deg = goal_reached_threshold_deg
        self.goal_reached_threshold_gripper_diff = goal_reached_threshold_gripper_diff
        self.gripper_command_delay_steps = gripper_command_delay_steps
        self.reset()

        self.model = DiffuserActor(
            feature_type=FeatureExtractorType(feature_type),
            embedding_dim=embedding_dim,
            fps_subsampling_factor=fps_subsampling_factor,
            # data-derived, not config: set via set_workspace_bounds from the
            # dataset at training start; checkpoints persist it (buffer)
            workspace_bounds=None,
            rotation_parametrization=rotation_parametrization,
            quaternion_format=quaternion_format,
            diffusion_timesteps=diffusion_timesteps,
            nhist=num_history,
            ngrippers=1,
            prediction_horizon=prediction_horizon,
            relative=0,
            data_type=DataType.MESH,
            use_fps=use_fps,
            encode_openness=1,
            loss_weights=LossWeights(
                pos_loss=pos_loss_weight,
                rot_loss=rot_loss_weight,
                gripper_loss=gripper_loss_weight,
            ),
            predict_head_yaw=False,
            encoder_dropout=encoder_dropout,
            diffusion_dropout=diffusion_dropout,
            predictor_dropout=predictor_dropout,
        )

    def apply_torch_compile(self, mode: str = 'default'):
        # DiffuserActor's furthest-point-sampling loop is data-dependent and traces
        # poorly under torch.compile, so compilation is deliberately unsupported here.
        raise NotImplementedError(
            "MindmapDP3 (DiffuserActor) does not support torch.compile: its FPS loop is "
            "data-dependent and traces poorly. Keep training.use_torch_compile=False."
        )

    def set_workspace_bounds(self, workspace_bounds):
        """Set the data-derived normalization bounds (from the dataset's zarr
        attr). Stored in a registered buffer, so checkpoints persist them and
        eval-time loading restores the TRAINING bounds automatically."""
        bounds = torch.as_tensor(workspace_bounds, dtype=torch.float32)
        assert bounds.shape == (2, 3), f"expected (2,3) bounds, got {tuple(bounds.shape)}"
        self.model.workspace_bounds = bounds.to(self.model.workspace_bounds.device)

    def _model_inputs(self, obs: Dict[str, torch.Tensor]):
        # buffer moves with .to(device); all-zeros means it was never set
        assert bool(torch.any(self.model.workspace_bounds != 0)), (
            "workspace_bounds not set: call set_workspace_bounds(dataset.workspace_bounds) "
            "for training, or load a checkpoint that contains them for eval."
        )
        return (
            obs["vertex_features"],
            obs["vertices"],
            obs["vertices_valid_mask"],
            obs["gripper_history"].unsqueeze(-2),  # (B, nhist, 1, 8)
        )

    def compute_loss(self, batch):
        feats, verts, valid, hist = self._model_inputs(batch["obs"])
        gt = batch["action"].unsqueeze(-2)  # (B, npred, 1, 8)
        losses, _, _ = self.model(
            gt, None, None, None, None, feats, verts, valid, None, hist
        )
        total, pos, rot, grip, _ = losses
        loss_dict = {
            "total_loss": total.detach().item(),
            "pos_loss": float(pos),
            "rot_loss": float(rot),
            "gripper_loss": float(grip),
        }
        return total, loss_dict

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "gripper_history" in obs_dict:
            # Offline path (train-time action-MSE eval): history comes from the dataset.
            feats, verts, valid, hist = self._model_inputs(obs_dict)
            pred, _, _, _, _ = self.model(
                None, None, None, None, None, feats, verts, valid, None, hist,
                run_inference=True,
            )
            action = pred.squeeze(-2)  # (B, npred, 8) absolute [pos3, quat4, openness]
            return {"action": action, "action_pred": action}
        return self._predict_action_closed_loop(obs_dict)

    def _predict_action_closed_loop(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Closed-loop path (env runner, n_action_steps=1 -> called every control
        step). Replicates mindmap's CLOSED_LOOP_WAIT loop: keep commanding the
        current absolute goal; run a new diffusion inference only when the goal
        is reached or max_num_steps_to_goal is exceeded. The nvblox map updates
        every step regardless (inside the obs wrapper); the gripper-state history
        holds the states at past inference times."""
        assert obs_dict["agent_proprio"].shape[0] == 1, "closed loop supports batch size 1"
        current_state = self._current_policy_state(obs_dict["agent_proprio"][:, -1])

        self._steps_to_reach_goal += 1
        goal_reached = self._current_goal is not None and self._is_goal_reached(current_state)
        goal_timeout = self._steps_to_reach_goal > self.max_num_steps_to_goal

        if self._current_goal is None or goal_reached or goal_timeout:
            self._update_gripper_state_history(current_state)
            eval_obs = {
                "vertices": obs_dict["vertices"][:, -1],
                "vertex_features": obs_dict["vertex_features"][:, -1],
                "vertices_valid_mask": obs_dict["vertices_valid_mask"][:, -1],
                "gripper_history": torch.stack(self._eval_gripper_state_history, dim=1),
            }
            feats, verts, valid, hist = self._model_inputs(eval_obs)
            pred, _, _, _, _ = self.model(
                None, None, None, None, None, feats, verts, valid, None, hist,
                run_inference=True,
            )
            keypose = pred[:, 0, 0]  # (B, 8) absolute [pos3, quat_wxyz4, openness_logit]
            self._current_goal = torch.cat(
                [
                    keypose[:, 0:3],
                    torch.nn.functional.normalize(keypose[:, 3:7], dim=-1),
                    (keypose[:, 7:8] > 0.0).to(keypose.dtype),  # BCE logit -> binary closedness
                ],
                dim=-1,
            )
            self._steps_to_reach_goal = 0

        goal = self._current_goal
        gripper_command = torch.where(
            goal[:, 7:8] > 0.5,
            torch.full_like(goal[:, 7:8], self.gripper_close_action),
            torch.full_like(goal[:, 7:8], self.gripper_open_action),
        )
        self._last_commanded_closedness = float(goal[0, 7])
        env_action = torch.cat([goal[:, 0:3], goal[:, 3:7], gripper_command], dim=-1)
        return {"action": env_action.unsqueeze(1), "action_pred": goal.unsqueeze(1)}

    def _current_policy_state(self, agent_proprio: torch.Tensor) -> torch.Tensor:
        """agent_proprio (B, 9) = [tcp pos3, tcp quat4, jaw positions2] ->
        policy state (B, 8) = [pos3, quat4, closedness]. Closedness comes from the
        delay-based estimator over past commands (mindmap's online estimator),
        initialized from the jaw positions on the first step of an episode."""
        jaw_positions = agent_proprio[0, 7:9]
        if self._gripper_state_estimator is None:
            initially_closed = bool(
                (jaw_positions[0] < GRIPPER_OPEN_THRESHOLD)
                and (jaw_positions[1] < GRIPPER_OPEN_THRESHOLD)
            )
            self._gripper_state_estimator = DelayBasedGripperStateEstimator(
                initial_state=initially_closed,
                steps_commanded_to_take_affect=self.gripper_command_delay_steps,
            )
        self._gripper_state_estimator.update(self._last_commanded_closedness)
        current_closedness = torch.tensor(
            [[float(self._gripper_state_estimator.get_state())]],
            dtype=agent_proprio.dtype, device=agent_proprio.device,
        )
        return torch.cat(
            [agent_proprio[:, 0:3], agent_proprio[:, 3:7], current_closedness], dim=-1
        )

    def _is_goal_reached(self, current_state: torch.Tensor) -> bool:
        """mindmap's ArmEmbodiment.is_goal_reached: position, rotation and
        gripper-closedness thresholds (their get_error_to_goal math)."""
        goal = self._current_goal[0]
        current = current_state[0]
        position_error_m = torch.norm(current[0:3] - goal[0:3])
        relative_quaternion = quaternion_multiply(
            quaternion_invert(current[3:7]), goal[3:7]
        )
        rotation_error_deg = math.degrees(
            torch.norm(quaternion_to_axis_angle(relative_quaternion))
        )
        gripper_difference = torch.abs(goal[7] - current[7])
        return bool(
            position_error_m < self.goal_reached_threshold_m
            and rotation_error_deg < self.goal_reached_threshold_deg
            and gripper_difference < self.goal_reached_threshold_gripper_diff
        )

    def _update_gripper_state_history(self, current_state: torch.Tensor):
        """Append the current state at inference time; the first inference fills
        the whole history with it (mindmap's _update_gripper_history)."""
        if self._eval_gripper_state_history is None:
            self._eval_gripper_state_history = [current_state] * self.num_history
        else:
            self._eval_gripper_state_history.append(current_state)
            self._eval_gripper_state_history = self._eval_gripper_state_history[-self.num_history:]

    def set_normalizer(self, normalizer: LinearNormalizer):
        # Identity by contract; DiffuserActor normalizes internally.
        self._normalizer = normalizer

    def reset(self):
        self._eval_gripper_state_history = None
        self._current_goal = None
        self._steps_to_reach_goal = 0
        self._gripper_state_estimator = None
        self._last_commanded_closedness = None

    def get_optimizer_param_groups(self, weight_decay: float):
        """mindmap Trainer.get_optimizer: parameters whose name contains 'bias',
        'LayerNorm.weight' or 'LayerNorm.bias' get no weight decay (verbatim
        filter, including its limitation that torch LayerNorm submodules named
        e.g. 'norm.weight' do NOT match)."""
        no_decay_name_parts = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        no_decay_parameters, decay_parameters = [], []
        for parameter_name, parameter in self.named_parameters():
            if any(part in parameter_name for part in no_decay_name_parts):
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)
        return [
            {"params": no_decay_parameters, "weight_decay": 0.0},
            {"params": decay_parameters, "weight_decay": weight_decay},
        ]
