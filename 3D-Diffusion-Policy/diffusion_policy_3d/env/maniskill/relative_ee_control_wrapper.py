import gym
import torch
import numpy as np
import copy
from diffusion_policy_3d.env.maniskill.utils import pose9d_to_mat, mat_to_pose6d_euler, pose7_to_mat, mat_to_pose9d


class RelativeEEControlWrapper(gym.Wrapper):
    """
    Transforms the stacked history into relative EE space based on the latest step's TCP pose (anchor).
    Transforms policy's relative EE targets back into absolute EE targets for pd_ee_pose control.
    """
    def __init__(self, env):
        super().__init__(env)
        self.T_cur_anchor_tcp_to_base = None
        self.last_env_actions = None  

    def step(self, action):
        """
        Action Transfrom -> Env Step -> Observation Transform
        """
            
        env_action = self._transform_action(action)
        self.last_env_actions = env_action.detach().clone()     # saved for logging
        obs, reward, done, info = self.env.step(env_action)
        rel_obs = self._transform_obs(obs)

        return rel_obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        return self._transform_obs(obs)

    def _transform_obs(self, obs):
        """
        Takes the raw obs dictionary (with stacked n_obs_steps sequences) and transforms
        proprioception and Gaussians into the anchor frame (the TCP pose of the current timestep).
        """
        tcp_pose_obs = obs['agent_proprio'][..., :-2]   # (n_obs_steps, 7), position + quaternion
        
        # Anchor frame: TCP at the last observation step
        anchor_tcp_to_base = pose7_to_mat(tcp_pose_obs[-1])  # (4, 4)
        self.T_cur_anchor_tcp_to_base = anchor_tcp_to_base.clone()
        anchor_base_to_tcp = torch.linalg.inv(anchor_tcp_to_base)
        R_anchor_base_to_tcp = anchor_base_to_tcp[:3, :3]
        t_anchor_base_to_tcp = anchor_base_to_tcp[:3, 3]

        # Transform obs TCP poses into anchor TCP
        tcp_to_base = pose7_to_mat(tcp_pose_obs)
        relative_tcp = anchor_base_to_tcp.unsqueeze(0) @ tcp_to_base
        
        rel_obs = copy.copy(obs)

        # Transform proprioception: relative EE pose + gripper
        rel_tcp_pose9d = mat_to_pose9d(relative_tcp)
        gripper_state = obs['agent_proprio'][..., -2:]      # absolute gripper state
        rel_obs['agent_proprio'] = torch.cat([rel_tcp_pose9d, gripper_state], dim=-1)

        # Transform positions: p_anchor = R_rel @ p_ee(i) + t_rel
        if 'gs_positions' in rel_obs:
            pos = rel_obs['gs_positions']
            rel_obs['gs_positions'] = pos @ R_anchor_base_to_tcp.T + t_anchor_base_to_tcp
            
        # Transform 9D rotations: R_anchor = R_rel @ R_ee(i)
        if 'gs_rotations_9d' in rel_obs:
            rot9d = rel_obs['gs_rotations_9d']
            T_obs, N = rot9d.shape[:2]
            rot_mats = rot9d.view(T_obs, N, 3, 3)
            rel_obs['gs_rotations_9d'] = (R_anchor_base_to_tcp @ rot_mats).view(T_obs, N, 9)
            
        # Transform surface normals: n_anchor = R_rel @ n_ee(i)
        if 'gs_surface_normals' in rel_obs:
            normals = rel_obs['gs_surface_normals']
            rel_obs['gs_surface_normals'] = normals @ R_anchor_base_to_tcp.T
        
        if 'gs_log_scales' in rel_obs:
            # TODO (currently not used during encoding)
            pass
        
        return rel_obs

    def _transform_action(self, action):
        """
        Transform relative EE action in anchor tcp frame to absolute EE action in base frame
        action: (n_action_steps, 10) -> [9D rel pose, 1D gripper]
        """

        rel_action_pose9d = action[..., :9]
        gripper_action = action[..., 9:]
        
        # Transform relative EE action to absolute EE action using the anchor frame
        T_rel_action = pose9d_to_mat(rel_action_pose9d)
        T_target = self.T_cur_anchor_tcp_to_base.unsqueeze(0) @ T_rel_action
        abs_pose6 = mat_to_pose6d_euler(T_target)
        
        # Re-attach gripper action
        env_action = torch.cat([abs_pose6, gripper_action], dim=-1)

        return env_action
