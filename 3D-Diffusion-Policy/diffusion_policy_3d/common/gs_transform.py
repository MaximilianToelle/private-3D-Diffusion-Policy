import time
import torch
import os
import re
import gymnasium as gym

# Import GSWorld utilities
from gsworld.utils.gs_utils import transform_gaussians, transform_batched_gaussians
from gsworld.utils.pcd_utils import extract_rigid_transform
from gsworld.utils.gaussian_merger import GaussianModelMerger
from gsworld.constants import (
    fr3_gs_semantics, 
    sim2gs_arm_trans, 
    obj_gs_semantics, 
    robot_scan_qpos, 
    object_offset,
    sim2gs_object_transforms,
    object_scale,
    CFG_DIR
)
from mani_skill.utils.structs.pose import Pose


class GSTransform:
    """
    Utility class to transform the 17-channel 3D Gaussian Splatting representation
    (xyz, rotation, scale, opacity, color, and surface normals) to robot `qpos` 
    and `actor_poses`.
    """
    def __init__(
        self, 
        scene_gs_cfg_name="fr3_stack", 
        robot_uid="fr3_umi_wrist435", 
        device="cuda",
        max_parallel_envs=1,
    ):
        self.device = torch.device(device)
        self.scene_gs_cfg_name = scene_gs_cfg_name
        self.max_parallel_envs = max_parallel_envs

        # Constants from GSWorld
        self.dict_map_fr3_link_name_gsworld_semantic_id = fr3_gs_semantics
        self.sim2gs_arm_trans = torch.tensor(sim2gs_arm_trans, dtype=torch.float32, device=self.device)
        self.dict_map_all_obj_names_gsworld_semantic_ids = obj_gs_semantics
        
        # Load the initial, unified gaussian splatting scene model 
        merger = GaussianModelMerger()
        path = os.path.join(CFG_DIR, f"{scene_gs_cfg_name}.json")
        merger.load_models_from_config(path)
        self.init_gs_scene_model = merger.merge_models()

        # Temporary ManiSkill environment to compute Forward Kinematics (FK)
        # Instead of implementing FK entirely by hand, we use the SAPIEN engine.
        # We create num_envs=max_parallel_envs so we can batch FK across multiple timesteps
        # -> eliminating the sequential T loop in transform_sequence
        env4fk = gym.make(
            "Empty-v1",
            obs_mode="none",
            reward_mode="none",
            enable_shadow=False,
            robot_uids=robot_uid,
            render_mode=None,
            num_envs=max_parallel_envs,
            sim_config=dict(sim_freq=100, control_freq=20),
            sim_backend="gpu",
        )
        env4fk.reset(seed=0)
        self.env4fk = env4fk.unwrapped
        
        # setting the robot to its GS scan pose, 
        # and caching the initial transformation matrices of the links in the GS scan pose wrt to sapien simulation base frame
        # -> later needed for GS transform calculation
        
        self.env4fk.agent.robot.set_qpos(robot_scan_qpos[robot_uid])
        self.transform_local_robot_gs_scan_pose_to_sapien_base_frame = []
        for link in self.env4fk.agent.robot.get_links():
            # With num_envs>1, to_transformation_matrix() returns (num_envs, 4, 4). 
            # All envs have the same scan pose, so we take [0].
            link_mat = link.pose.to_transformation_matrix()[0].to(self.device)
            self.transform_local_robot_gs_scan_pose_to_sapien_base_frame.append(link_mat)

        # Cache semantic link_specific_gs_scene_indices of each Gaussian
        gs_scene_gsworld_semantic_ids = self.init_gs_scene_model._semantics.long().squeeze(-1).contiguous()
        self._dict_map_name_gs_scene_indices = {} # key -> index tensor for transform_gaussians
        
        # Robot
        for link in self.env4fk.agent.robot.get_links():
            if link.name in self.dict_map_fr3_link_name_gsworld_semantic_id:
                if link.name == "fr3_hand": # fr3_hand is a list of semantic_indices
                    semantic_target_id = torch.tensor(self.dict_map_fr3_link_name_gsworld_semantic_id[link.name], device=self.device, dtype=torch.long)
                    self._dict_map_name_gs_scene_indices[link.name] = torch.where(torch.isin(gs_scene_gsworld_semantic_ids, semantic_target_id))[0]
                else:
                    semantic_target_id = self.dict_map_fr3_link_name_gsworld_semantic_id[link.name]
                    self._dict_map_name_gs_scene_indices[link.name] = torch.where(gs_scene_gsworld_semantic_ids == semantic_target_id)[0]

        # Objects
        for actor_key, sem_id in self.dict_map_all_obj_names_gsworld_semantic_ids.items():
            self._dict_map_name_gs_scene_indices[actor_key] = torch.where(gs_scene_gsworld_semantic_ids == sem_id)[0]

        # Remove all keys with empty vectors (i.e. objects not present in this scene's GS model)
        self._dict_map_name_gs_scene_indices = {k: v for k, v in self._dict_map_name_gs_scene_indices.items() if len(v) > 0}
        
        # All actor names in the scene 
        self._all_actor_names_in_scene = set(k for k in self._dict_map_name_gs_scene_indices if k in self.dict_map_all_obj_names_gsworld_semantic_ids)

        # Performance optimization: cache all moving link/actor specific gs_scene_indices and object mapping
        # NOTE: we compute the rigid transform for each moving object and 
        # apply it via object id to all Gaussians belonging to an object
        all_object_id_ordered_gs_scene_indices = []
        all_object_id_ordered_indices_to_object_ids = []
        self._moving_objects = [] # list of dicts with metadata
        moving_object_idx = 0
        
        # 1. Robot Links
        robot_links = self.env4fk.agent.robot.get_links()
        for i, link in enumerate(robot_links):
            if link.name in self._dict_map_name_gs_scene_indices:
                link_specific_gs_scene_indices = self._dict_map_name_gs_scene_indices[link.name]
                all_object_id_ordered_gs_scene_indices.append(link_specific_gs_scene_indices)
                all_object_id_ordered_indices_to_object_ids.append(torch.full((link_specific_gs_scene_indices.shape[0],), moving_object_idx, device=self.device, dtype=torch.long))
                self._moving_objects.append({
                    "type": "robot",
                    "name": link.name,
                    "link_idx": i
                })
                moving_object_idx += 1
                
        # 2. Actors (Objects)
        for actor_key in self.dict_map_all_obj_names_gsworld_semantic_ids:
            if actor_key in self._dict_map_name_gs_scene_indices:
                actor_specific_gs_scene_indices = self._dict_map_name_gs_scene_indices[actor_key]
                all_object_id_ordered_gs_scene_indices.append(actor_specific_gs_scene_indices)
                all_object_id_ordered_indices_to_object_ids.append(torch.full((actor_specific_gs_scene_indices.shape[0],), moving_object_idx, device=self.device, dtype=torch.long))
                self._moving_objects.append({
                    "type": "actor",
                    "name": actor_key
                })
                moving_object_idx += 1
                
        self._all_object_id_ordered_gs_scene_indices = torch.cat(all_object_id_ordered_gs_scene_indices)
        self._all_object_id_ordered_indices_to_object_ids = torch.cat(all_object_id_ordered_indices_to_object_ids)
         
        # Pre-cache inverses and transforms
        self.transform_sapien_base_frame_to_local_robot_gs_scan_pose = torch.stack([torch.linalg.inv(m) for m in self.transform_local_robot_gs_scan_pose_to_sapien_base_frame])
        self.gs2sim_arm_trans = torch.linalg.inv(self.sim2gs_arm_trans)

        # Pre-calculate actor scales and inverse transforms ONLY for scene-relevant actors
        # (NOTE: rigid body assumption + uniform scale assumption!)
        self.gs_actor_scale_wrt_gs_robot_arm_scale = {}     # Used for matrix division
        self.gs_actor_scale_wrt_sapien = {}                 # Used for GS rendering
        self.gs2sim_actor_trans = {}                        # Pre-cached inverse transforms
        for actor_key in self._all_actor_names_in_scene:
            if actor_key not in sim2gs_object_transforms:
                raise ValueError("extracted actor key does not have a sim2gs object transform")
            sim2gs_actor_trans = torch.tensor(sim2gs_object_transforms[actor_key], device=self.device, dtype=torch.float32)
            self.gs2sim_actor_trans[actor_key] = sim2gs_actor_trans.inverse()
            
            gs_actor2gs_arm_trans = self.sim2gs_arm_trans @ torch.eye(4, device=self.device) @ self.gs2sim_actor_trans[actor_key]
            _, scale, _, _ = extract_rigid_transform(gs_actor2gs_arm_trans)
            
            self.gs_actor_scale_wrt_gs_robot_arm_scale[actor_key] = scale
            self.gs_actor_scale_wrt_sapien[actor_key] = scale * object_scale[actor_key]


    def transform(self, qpos: torch.Tensor, actor_poses: dict):
        """
        Transforms the 3D Gaussian cloud for a single timestep.
        qpos: (9,) tensor
        actor_poses: dict mapping actor_name -> (13,) tensor
        Returns:
        gsplats: dict representing the 3d gaussian representation
        """
        
        if qpos.dim() > 1:
            qpos = qpos.squeeze()
            
        # 1. Update robot kinematics
        self.env4fk.agent.robot.set_qpos(qpos)
        
        # 2. Collect the transforms of all moving parts (robot links and objects) 
        K = len(self._moving_objects)
        batch_rot = torch.zeros((K, 3, 3), device=self.device)
        batch_trans = torch.zeros((K, 3), device=self.device)
        batch_scale = torch.ones(K, device=self.device)
        
        robot_links = self.env4fk.agent.robot.get_links()
        
        for i, part in enumerate(self._moving_objects):
            if part["type"] == "robot":
                link_idx = part["link_idx"]
                transform_robot_link_to_sapien_base_frame = robot_links[part["link_idx"]].pose.to_transformation_matrix()[0].to(self.device)
                
                # Note: Hardcoded xarm object offset
                if "xarm" in self.env4fk.agent.uid:
                    for j in range(3):
                        transform_robot_link_to_sapien_base_frame[j, 3] += object_offset["xarm_arm"][j]
                
                # Compute T_sim2gs
                T = self.sim2gs_arm_trans @ transform_robot_link_to_sapien_base_frame @ self.transform_sapien_base_frame_to_local_robot_gs_scan_pose[link_idx] @ self.gs2sim_arm_trans
                batch_rot[i] = T[:3, :3]
                batch_trans[i] = T[:3, 3]
                
            elif part["type"] == "actor":
                actor_key = part["name"]
                if actor_key not in actor_poses or actor_key not in self.gs2sim_actor_trans:
                    raise ValueError("actor_key not known!")
                    
                full_pose = actor_poses[actor_key]
                pose_tensor = full_pose[:7].to(self.device).view(-1) # Extract position and quaternion
                
                # SAPIEN Pose.create expects batched input
                mat = Pose.create(pose=pose_tensor.unsqueeze(0)).to_transformation_matrix()[0]
                
                if actor_key in object_offset:
                    for j in range(3):
                        mat[j, 3] += object_offset[actor_key][j]
                        
                T = self.sim2gs_arm_trans @ mat @ self.gs2sim_actor_trans[actor_key]
                
                batch_scale[i] = self.gs_actor_scale_wrt_sapien[actor_key]
                batch_rot[i] = T[:3, :3] / self.gs_actor_scale_wrt_gs_robot_arm_scale[actor_key] 
                batch_trans[i] = T[:3, 3]
            
        # Expand transforms to active gaussians
        final_rot = batch_rot[self._all_object_id_ordered_indices_to_object_ids]     # (N, 3, 3)
        final_trans = batch_trans[self._all_object_id_ordered_indices_to_object_ids] # (N, 3)
        final_scale = batch_scale[self._all_object_id_ordered_indices_to_object_ids] # (N,)
        
        # 3. Apply Transformation to all active Gaussians
        gs_xyz, gs_scaling, gs_rotation, gs_opacity, gs_surf_normals = transform_gaussians(
            self.init_gs_scene_model,
            selected_indices=self._all_object_id_ordered_gs_scene_indices,
            scale=final_scale,
            rot_mat=final_rot,
            translation=final_trans,
            new_opacity=None,
        )
        
        # Assemble channels natively
        opacity = gs_opacity.view(-1, 1)
        f_dc = self.init_gs_scene_model._features_dc[self._all_object_id_ordered_gs_scene_indices].view(-1, 3)
            
        gsplats = {
            "positions": gs_xyz,            
            "quaternions": gs_rotation,     
            "log_scales": gs_scaling,       
            "logit_opacities": opacity,     
            "features_dc": f_dc,
            "surface_normals": gs_surf_normals,                   
        }
        
        return gsplats


    def transform_sequence(self, agent_pos: torch.Tensor, actor_poses: dict):
        """
        Batched Sequence Transformation.
        agent_pos: (T, 9) CPU
        actor_poses: dict mapping actor_name -> (T, 7) CPU tensor
        """
        T_seq = agent_pos.shape[0]
        assert T_seq <= self.max_parallel_envs, (
            f"Sequence length {T_seq} exceeds max_parallel_envs={self.max_parallel_envs}. "
            f"Increase max_parallel_envs in GSTransform.__init__."
        )
        K = len(self._moving_objects)
        
        batch_rot = torch.zeros((T_seq, K, 3, 3), device=self.device)
        batch_trans = torch.zeros((T_seq, K, 3), device=self.device)
        batch_scale = torch.ones((T_seq, K), device=self.device)
        
        robot_links = self.env4fk.agent.robot.get_links()
        
        # 1. Batch FK: set all T qpos at once across parallel envs, read all link poses in one shot.
        # Pad to max_parallel_envs since the env was created with num_envs=max_parallel_envs.
        qpos_dim = agent_pos.shape[1]
        padded_qpos = torch.zeros((self.max_parallel_envs, qpos_dim), device=self.device)
        padded_qpos[:T_seq] = agent_pos.to(self.device)
        self.env4fk.agent.robot.set_qpos(padded_qpos)
        
        transforms_robot_links_to_sapien_base_frame = torch.zeros((T_seq, len(robot_links), 4, 4), device=self.device)
        for i, link in enumerate(robot_links):
            # to_transformation_matrix() returns (num_envs, 4, 4); slice to T_seq
            transforms_robot_links_to_sapien_base_frame[:, i] = link.pose.to_transformation_matrix()[:T_seq]
                
        # Compute rigid kinematics fully vectorized
        # going through all moving objects where i corresponds to object_id  
        for i, part in enumerate(self._moving_objects):
            if part["type"] == "robot":
                link_idx = part["link_idx"]
                transform_robot_link_to_sapien_base_frame = transforms_robot_links_to_sapien_base_frame[:, link_idx].clone() # (T_seq, 4, 4)
                
                if "xarm" in self.env4fk.agent.uid:
                    for j in range(3):
                        transform_robot_link_to_sapien_base_frame[:, j, 3] += object_offset["xarm_arm"][j]

                # gs to sapien base frame -> sapien to local robot link frame -> fk transform wrt sapien base frame -> sapien base frame 2 gs           
                # NOTE: whatever scale is baked into gs2sim_arm_trans gets perfectly cancelled by its inverse sim2gs_arm_trans
                T_mat = self.sim2gs_arm_trans.unsqueeze(0) @ transform_robot_link_to_sapien_base_frame @ self.transform_sapien_base_frame_to_local_robot_gs_scan_pose[link_idx].unsqueeze(0) @ self.gs2sim_arm_trans.unsqueeze(0)
                batch_rot[:, i] = T_mat[:, :3, :3]
                batch_trans[:, i] = T_mat[:, :3, 3]
                
            elif part["type"] == "actor":
                actor_key = part["name"]
                if actor_key not in actor_poses or actor_key not in self.gs2sim_actor_trans:
                    raise ValueError("actor_key not known!")
                    
                full_pose = actor_poses[actor_key].to(self.device) # (T_seq, 7)
                
                pose_tensor = full_pose[:, :7].to(self.device)
                new_sim_transfrom_matrix = Pose.create(pose=pose_tensor).to_transformation_matrix()
                
                if actor_key in object_offset:
                    for j in range(3):
                        new_sim_transfrom_matrix[:, j, 3] += object_offset[actor_key][j]

                # Object's GS frame → SAPIEN world -> transfrom within sapien world frame -> transform back to GS rendering frame        
                T_mat = self.sim2gs_arm_trans.unsqueeze(0) @ new_sim_transfrom_matrix @ self.gs2sim_actor_trans[actor_key].unsqueeze(0)
                
                # gs2sim_actor_trans is different from sim2gs_arm_trans, so scale does not cancel out!
                # See init to understand gs_actor_scale_wrt_sapien and gs_actor_scale_wrt_gs_robot_arm_scale
                batch_scale[:, i] = self.gs_actor_scale_wrt_sapien[actor_key]
                batch_rot[:, i] = T_mat[:, :3, :3] / self.gs_actor_scale_wrt_gs_robot_arm_scale[actor_key]
                batch_trans[:, i] = T_mat[:, :3, 3]
                
        final_rot = batch_rot[:, self._all_object_id_ordered_indices_to_object_ids]     # (T_seq, N, 3, 3)
        final_trans = batch_trans[:, self._all_object_id_ordered_indices_to_object_ids] # (T_seq, N, 3)
        final_scale = batch_scale[:, self._all_object_id_ordered_indices_to_object_ids] # (T_seq, N)
        
        # Fully batched transform over sequence length 
        gs_xyz, gs_scaling, gs_rotation, gs_opacity, gs_surf_normals = transform_batched_gaussians(
            self.init_gs_scene_model,
            selected_indices=self._all_object_id_ordered_gs_scene_indices,
            scale=final_scale,
            rot_mat=final_rot,
            translation=final_trans,
        )
            
        opacity = gs_opacity.view(T_seq, -1, 1)
        
        # features_dc is unchanged across time, broadcast to (T_seq, N, 3)
        f_dc = self.init_gs_scene_model._features_dc[self._all_object_id_ordered_gs_scene_indices].view(-1, 3)
        f_dc = f_dc.unsqueeze(0).expand(T_seq, -1, -1)
                
        return {
            "positions": gs_xyz,            
            "quaternions": gs_rotation,     
            "log_scales": gs_scaling,       
            "logit_opacities": opacity,     
            "features_dc": f_dc,
            "surface_normals": gs_surf_normals,                   
        }
