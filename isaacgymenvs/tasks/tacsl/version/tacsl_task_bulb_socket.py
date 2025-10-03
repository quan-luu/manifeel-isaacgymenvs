# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: Class for nut-bolt screw task.

Inherits nut-bolt environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskNutBoltScrew

Initial Franka/nut states are ideal for M16 nut-and-bolt.
In this example, initial state randomization is not applied; thus, policy should succeed almost instantly.
"""

import hydra
import math
import omegaconf
import os
import torch
import numpy as np

from isaacgym import gymapi, gymtorch
from isaacgymenvs.utils import torch_jit_utils as torch_utils
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
from isaacgymenvs.tasks.tacsl.tacsl_env_bulb_socket import TacSLEnvBulbSocketScrew
from isaacgymenvs.tasks.tacsl.tacsl_task_image_augmentation import TacSLTaskImageAugmentation
from isaacgymenvs.utils import torch_jit_utils



class TacSLTaskBulbSocketScrew(TacSLTaskImageAugmentation, TacSLEnvBulbSocketScrew, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize task superclass."""

        self.cfg = cfg
        self._get_task_yaml_params()

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self._acquire_task_tensors()

        if self.cfg_task.env.use_isaac_gym_tactile:
            assert self.cfg_task.env.use_gelsight, "shear force currently works only with gelsight fingers"
            # Open finger to render nominal tactile sensor
            self.initialize_franka_robot_open_hand()
            # Initialize tactile sensors
            self.initialize_tactile_rgb_camera()

        if self.cfg_task.env.use_shear_force:
            assert self.cfg_task.env.use_gelsight, "shear force currently works only with gelsight fingers"
            num_divs = [self.cfg_task.env.num_shear_rows, self.cfg_task.env.num_shear_cols]
            self.initialize_penalty_based_tactile(num_divs=num_divs)

        if self.cfg_task.env.task_type == 'placement':
            # The placement task moves the peg to the tip of the placement pad, no insertion
            self.cfg_task.rl.insertion_frac = 0.0

        if self.viewer is not None:
            self._set_viewer_params()

        if self.cfg_base.mode.export_scene:
            self.export_scene(label='tacsl_task_screw')

        self.set_friction_damping_params(joint_friction=self.cfg_task.env.joint_friction,
                                         joint_damping=self.cfg_task.env.joint_damping)

        self.image_obs_keys = [k for k, v in self.obs_dims.items() if len(v) > 2 and 'force_field' not in k]
        self.init_image_augmentation()
        
        self.reset_idx(torch.arange(self.num_envs))


    def initialize_franka_robot_open_hand(self):
        """
        Initialize Franka robot to default dof position.
        """
        self.dof_pos[:, 0:7] = torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device)
        self.dof_pos[:, 7:] = self.cfg_task.env.get("franka_open_gripper_width",
                                                    self.asset_info_franka_table.franka_gripper_width_max)

        self.ctrl_target_dof_pos[:] = self.dof_pos[:]
        self.dof_vel[:, 0:self.franka_num_dofs] = 0.0

        franka_actor_ids_sim_int32 = self.actor_ids_sim_tensors['franka'].to(dtype=torch.int32, device=self.device)[:]
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(franka_actor_ids_sim_int32),
                                              len(franka_actor_ids_sim_int32))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.refresh_all_tensors()

    def set_friction_damping_params(self, joint_friction=None, joint_damping=None):
        """
        Set friction and damping parameters for the robot joints.

        Args:
            joint_friction: Friction values for the joints.
            joint_damping: Damping values for the joints.
        """
        if joint_friction is None and joint_damping is None:
            return

        for env_id in range(self.num_envs):
            env_ptr, franka_handle = self.env_ptrs[env_id], self.actor_handles['franka']

            franka_dof_props = self.gym.get_actor_dof_properties(env_ptr, franka_handle)

            if joint_friction is not None:
                franka_dof_props['friction'][:9] = joint_friction[:9]
            if joint_damping is not None:
                franka_dof_props['damping'][:9] = joint_damping[:9]

            self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

    def get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets
    
    def _acquire_task_tensors(self):
        """Acquire tensors for bulb insertion task."""

        # Bulb-socket origin tensors (local frame)
        # self.bulb_keypoint_origin_local = torch.zeros((self.num_envs, 3), device=self.device)
        # self.socket_keypoint_origin_local = torch.zeros((self.num_envs, 3), device=self.device)
        self.bulb_keypoint_origin_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.socket_keypoint_origin_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        # Scale Z-axis for socket insertion height
        insertion_frac = self.cfg_task.rl.insertion_frac
        insertion_height = 1.0 - insertion_frac
        self.socket_keypoint_origin_local[:, 2] = self.socket_heights.squeeze(-1) * insertion_height

        # Socket tip position (full height)
        self.socket_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        self.socket_tip_pos_local[:, 2] = self.socket_heights.squeeze(-1)

        # Keypoint offsets and tensors
        num_keypoints = self.cfg_task.rl.num_keypoints
        self.keypoint_offsets = self.get_keypoint_offsets(num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_bulb = torch.zeros((self.num_envs, num_keypoints, 3), device=self.device)
        self.keypoints_socket = torch.zeros_like(self.keypoints_bulb, device=self.device)

        # Identity quaternion
        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(self.num_envs, 4)

        # Action tensors
        num_actions = self.cfg_task.env.numActions
        self._actions = torch.zeros((self.num_envs, num_actions), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, num_actions), device=self.device)

        # Target positions along Z-axis (full insertion)
        target_heights = self.asset_info_franka_table.table_height + self.socket_heights + self.bulb_lengths
        self.target_pos = target_heights.view(-1, 1) * torch.tensor([0.0, 0.0, 1.0], device=self.device)       
    
    # def _refresh_task_tensors(self):
    #     """Refresh tensors."""
    #     # Compute pos of keypoints on gripper, bulb, and socket in world frame
    #     for idx, keypoint_offset in enumerate(self.keypoint_offsets):
    #         self.keypoints_bulb[:, idx] = torch_jit_utils.tf_combine(self.bulb_quat,
    #                                                                  self.bulb_pos,
    #                                                                  self.identity_quat,
    #                                                                  (keypoint_offset + self.plug_keypoint_origin_local))[1]
    #         self.keypoints_socket[:, idx] = torch_jit_utils.tf_combine(self.socket_quat,
    #                                                                    self.socket_pos,
    #                                                                    self.identity_quat,
    #                                                                    (keypoint_offset + self.socket_keypoint_origin_local))[1]
    # def refresh_all_tensors(self):
    #     self.refresh_base_tensors()
    #     self.refresh_env_tensors()
    #     self._refresh_task_tensors()



    # def _get_task_yaml_params(self):
    #     """Initialize instance variables from YAML files."""

    #     cs = hydra.core.config_store.ConfigStore.instance()
    #     cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

    #     self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
    #     self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

    #     asset_info_path = '../../assets/factory/yaml/tacsl_asset_info_nut_bolt.yaml'  # relative to Gym's Hydra search path (cfg dir)
    #     self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
    #     self.asset_info_nut_bolt = self.asset_info_nut_bolt['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

    #     ppo_path = 'train/FactoryTaskNutBoltScrewPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
    #     self.cfg_ppo = hydra.compose(config_name=ppo_path)
    #     self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    # def _acquire_task_tensors(self):
    #     """Acquire tensors."""

    #     # target_heights = self.cfg_base.env.table_height + self.bolt_head_heights + self.nut_heights * 0.5
    #     target_heights = (self.cfg_base.env.table_height + self.socket_heights 
    #                       + self.bulb_lengths * 1.0   # multiply by 1.0 for full insertion or by 0.5 for half-insertion
    #                       )

    #     self.target_pos = target_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        self.fingerpad_midpoint_pos = fc.translate_along_local_z(pos=self.finger_midpoint_pos,
                                                                 quat=self.hand_quat,
                                                                 offset=self.asset_info_franka_table.franka_finger_length - self.asset_info_franka_table.franka_fingerpad_length * 0.5,
                                                                 device=self.device)
        # self.finger_nut_keypoint_dist = self._get_keypoint_dist(body='finger_nut')
        # self.nut_keypoint_dist = self._get_keypoint_dist(body='nut')
        # self.nut_dist_to_target = torch.norm(self.target_pos - self.nut_com_pos, p=2,
        #                                      dim=-1)  # distance between nut COM and target
        # self.nut_dist_to_fingerpads = torch.norm(self.fingerpad_midpoint_pos - self.nut_com_pos, p=2,
        #                                          dim=-1)  # distance between nut COM and midpoint between centers of fingerpads

        # keypoint distances
        self.finger_bulb_keypoint_dist = self._get_keypoint_dist(body='finger_bulb')
        self.bulb_keypoint_dist = self._get_keypoint_dist(body='bulb')

        # distance between bulb COM and target insertion position
        self.bulb_dist_to_target = torch.norm(
            self.target_pos - self.bulb_com_pos, p=2, dim=-1
        )

        # distance between bulb COM and midpoint of fingerpads (gripper center)
        self.bulb_dist_to_fingerpads = torch.norm(
            self.fingerpad_midpoint_pos - self.bulb_com_pos, p=2, dim=-1
        )

    def refresh_all_tensors(self):
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(actions=self.actions,
                                            ctrl_target_gripper_dof_pos=self.cfg_task.env.get("franka_close_gripper_width", 0.0),
                                            do_scale=True)

        sim_dt_noise = self.cfg_task.env.get("sim_dt_noise", 0)
        if sim_dt_noise > 0.0:
            sim_params = self.gym.get_sim_params(self.sim)
            sim_params.dt = self.cfg_base.sim.dt * (1 + torch.rand(1) * sim_dt_noise)
            self.gym.set_sim_params(self.sim, sim_params)

        num_extra_control_steps = self.cfg_task.env.get("num_additional_control_steps", 0)  # for backward compatibility
        num_additional_control_steps_noise = self.cfg_task.env.get("num_additional_control_steps_noise", 0)
        if num_additional_control_steps_noise:
            num_extra_control_steps += torch.randint(0,
                                                     self.cfg_task.env.num_additional_control_steps_noise + 1,
                                                     (1,)
                                                     )[0].item()
        self.execute_control_loop(num_extra_control_steps)

    def execute_control_loop(self, num_control_steps):
        """
        Execute the control loop for a specified number of steps.

        Args:
            num_control_steps: Number of control steps to execute.
        """
        for _ in range(num_control_steps):
            # execute previous control signal
            self.gym.simulate(self.sim)

            # refresh tensors
            self.refresh_all_tensors()

            # generate new control signal
            self.generate_ctrl_signals()

    def apply_screw_primitive_task_space(self):
        """Apply screw primitive in task space."""
        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_midpoint_pos.detach().clone()
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_midpoint_quat.detach().clone()

        # get screw motion as delta euler angle
        screw_motion = torch_utils.quat_from_euler_xyz(
            torch.tensor([0.0], device=self.device),
            torch.tensor([0.0], device=self.device),
            torch.tensor([np.pi/2], device=self.device)).expand(self.num_envs, 4)
        self.ctrl_target_fingertip_midpoint_pos[:, 2] -= 0.005  # move down while screwing
        self.ctrl_target_fingertip_midpoint_quat[:] = torch_utils.quat_mul(self.fingertip_midpoint_quat,
                                                                           screw_motion)

        self._move_to_target_pose_and_gripper_width(self.ctrl_target_fingertip_midpoint_pos,
                                                    self.ctrl_target_fingertip_midpoint_quat,
                                                    gripper_dof_pos=0.0, gentle_gripper_close=False,
                                                    sim_steps=self.cfg_task.env.num_gripper_close_sim_steps * 2)

    def apply_screw_primitive(self):
        """Apply screw primitive."""
        self.refresh_all_tensors()
        target_dof = self.dof_pos.clone()
        target_dof[:, 6] += 1.

        DEFAULT_K_GAINS = [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]
        DEFAULT_D_GAINS = [50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0]
        joint_prop_gains = torch.tensor(DEFAULT_K_GAINS, device=self.device) / 100.
        joint_deriv_gains = torch.tensor(DEFAULT_D_GAINS, device=self.device) / 100.

        # Step sim
        sim_steps = self.cfg_task.env.num_gripper_close_sim_steps * 4
        for _ in range(sim_steps):
            self.refresh_all_tensors()
            self.dof_torque[:, 0:7] = joint_prop_gains * (target_dof - self.dof_pos)[:, 0:7] + \
                                      joint_deriv_gains * (0.0 - self.dof_vel[:, 0:7])
            # keep prev torque applied by gripper
            self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.dof_torque),
                                                            gymtorch.unwrap_tensor(self.actor_ids_sim_tensors['franka']),
                                                            len(self.actor_ids_sim_tensors['franka']))
            self.render()
            self.gym.simulate(self.sim)

    def execute_terminal_primitive(self):
        """Execute terminal primitive actions."""
        if self.cfg_task.env.task_type == 'screwing':
            # do screw primitive
            self.apply_screw_primitive()

        if self.cfg_task.env.task_type in ['placement', 'screwing']:
            # open-gripper and lift
            self._open_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps//2)
            self._lift_gripper(gripper_dof_pos=0.1, sim_steps=self.cfg_task.env.num_gripper_close_sim_steps//2)


    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            self.execute_terminal_primitive()

        # self.refresh_base_tensors()
        # self.refresh_env_tensors()
        # self._refresh_task_tensors()
        self.refresh_all_tensors()
        self.compute_observations()
        self.compute_reward()

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            # Check if plug is at the goal location within the socket
            task_success = self._check_success()
            self.rew_buf[:] += task_success * self.cfg_task.rl.success_bonus
            self.extras['successes'] = torch.mean(task_success.float())

        self.prev_actions[:] = self._actions

        if self.cfg_base.mode.export_states:
            self.extract_poses()

    def compute_observations(self):
        """Compute observations."""

        if self.cfg_task.env.use_dict_obs:
            return self.compute_observations_dict_obs()

        return self.obs_buf  # shape = (num_envs, num_observations)

    def compute_observations_dict_obs(self):
        """
        Compute observations as a dictionary.

        Returns:
            obs_dict: Dictionary containing observations.
        """
        self.obs_dict['ee_pos'][:] = self.fingertip_midpoint_pos
        self.obs_dict['ee_quat'][:] = self.fingertip_midpoint_quat
        self.obs_dict['socket_pos'][:] = self.socket_pos
        self.obs_dict['socket_quat'][:] = self.socket_quat
        if 'bulb_pos' in self.cfg_task.env.obsDims or self.cfg_task.rl.asymmetric_observations:
            self.obs_dict['bulb_pos'][:] = self.bulb_pos
            self.obs_dict['bulb_quat'][:] = self.bulb_quat
        if 'eef_to_bulb_pos' in self.cfg_task.env.obsDims:
            eef_to_bulb_transform = torch_jit_utils.tf_combine(
                *torch_jit_utils.tf_inverse(self.fingertip_midpoint_quat, self.fingertip_midpoint_pos),
                self.bulb_quat, self.bulb_pos
            )
            self.obs_dict['eef_to_bulb_pos'][:] = eef_to_bulb_transform[1]
            self.obs_dict['eef_to_bulb_quat'][:] = eef_to_bulb_transform[0]
        if 'dof_pos' in self.cfg_task.env.obsDims or 'dof_pos' in self.cfg_task.env.stateDims:
            self.obs_dict['dof_pos'][:] = self.dof_pos
        if 'dof_vel' in self.cfg_task.env.obsDims or 'dof_vel' in self.cfg_task.env.stateDims:
            self.obs_dict['dof_vel'][:] = self.dof_vel
        if 'ee_lin_vel' in self.cfg_task.env.obsDims or self.cfg_task.rl.asymmetric_observations:
            self.obs_dict['ee_lin_vel'][:] = self.fingertip_midpoint_linvel
            self.obs_dict['ee_ang_vel'][:] = self.fingertip_midpoint_angvel
        if self.cfg_task.rl.add_contact_force_plug_decomposed or self.cfg_task.rl.add_contact_info_to_aac_states:
            self.obs_dict['bulb_socket_force'][:] = self.contact_force_pairwise[:, self.bulb_body_id_env, self.socket_body_id_env]
            if self.cfg_task.env.use_compliant_contact:
                self.obs_dict['bulb_left_elastomer_force'][:] = self.contact_force_pairwise[:, self.bulb_body_id_env, self.franka_body_ids_env['elastomer_left']]
                self.obs_dict['bulb_right_elastomer_force'][:] = self.contact_force_pairwise[:, self.bulb_body_id_env, self.franka_body_ids_env['elastomer_right']]
            else:
                self.obs_dict['bulb_left_elastomer_force'][:] = self.contact_force_pairwise[:, self.bulb_body_id_env, self.franka_body_ids_env['panda_leftfinger']]
                self.obs_dict['bulb_right_elastomer_force'][:] = self.contact_force_pairwise[:, self.bulb_body_id_env, self.franka_body_ids_env['panda_rightfinger']]
        if self.cfg_task.env.use_camera_obs:
            images = self.get_camera_image_tensors_dict()

            if self.cfg_task.env.use_isaac_gym_tactile:
                # Optionally subsample tactile image
                ssr = self.cfg_task.env.tactile_subsample_ratio
                for k in self.tactile_ig_keys:
                    images[k] = images[k][:, ::ssr, ::ssr]

            for cam in images:
                if cam in self.cfg_task.env.obsDims:
                    if images[cam].dtype == torch.uint8:
                        self.obs_dict[cam][..., :3] = images[cam] / 255.
                    else:
                        self.obs_dict[cam][..., :3] = images[cam]

            self.apply_image_augmentation_to_obs_dict()

        self.obs_dict['socket_pos'][:] = self.socket_pos + self.socket_obs_noise
        self.obs_dict['socket_pos_gt'][:] = self.socket_pos

        if self.cfg_task.env.use_shear_force:
            tactile_force_field_dict = self.get_tactile_force_field_tensors_dict()
            if self.cfg_task.env.use_tactile_field_obs:
                # for k in ['tactile_force_field_left', 'tactile_force_field_right']:
                for k in ['tactile_force_field_right']:
                    self.obs_dict[k][:] = tactile_force_field_dict[k]
                    if self.cfg_task.env.zero_out_normal_force_field_obs:
                        self.obs_dict[k][..., 0] *= 0.0
        # if self.cfg_task.rl.add_obs_finger_force:
            
        return self.obs_dict

    # def compute_observations(self):
    #     """Compute observations."""

    #     # Shallow copies of tensors
    #     obs_tensors = [self.fingertip_midpoint_pos,
    #                    self.fingertip_midpoint_quat,
    #                    self.fingertip_midpoint_linvel,
    #                    self.fingertip_midpoint_angvel,
    #                    self.bulb_com_pos,
    #                    self.bulb_com_quat,
    #                    self.bulb_com_linvel,
    #                    self.bulb_com_angvel]

    #     if self.cfg_task.rl.add_obs_finger_force:
    #         obs_tensors += [self.left_finger_force, self.right_finger_force]

    #     obs_tensors = torch.cat(obs_tensors, dim=-1)
    #     self.obs_buf[:, :obs_tensors.shape[-1]] = obs_tensors  # shape = (num_envs, num_observations)

    #     return self.obs_buf

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        # Get successful and failed envs at current timestep
        curr_successes = self._get_curr_successes()
        curr_failures = self._get_curr_failures(curr_successes)

        self._update_reset_buf(curr_successes, curr_failures)
        self._update_rew_buf(curr_successes)

    def _update_reset_buf(self, curr_successes, curr_failures):
        """Assign environments for reset if successful or failed."""

        self.reset_buf[:] = torch.logical_or(curr_successes, curr_failures)

    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep."""

        keypoint_reward = -(self.bulb_keypoint_dist + self.finger_bulb_keypoint_dist)
        action_penalty = torch.norm(self.actions, p=2, dim=-1)

        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale \
                          + curr_successes * self.cfg_task.rl.success_bonus

    # def reset_idx(self, env_ids):
    #     """Reset specified environments. Zero buffers."""

    #     self._reset_franka(env_ids)
    #     self._reset_object(env_ids)

    #     self._reset_buffers(env_ids)

    # def _reset_franka(self, env_ids):
    #     """Reset DOF states and DOF targets of Franka."""

    #     self.dof_pos[env_ids] = torch.cat((torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos,
    #                                                     device=self.device).repeat((len(env_ids), 1)),
    #                                        (self.bulb_widths_max[env_ids] * 0.5) * 1.1,  # buffer on gripper DOF pos to prevent initial contact
    #                                        (self.bulb_widths_max[env_ids] * 0.5) * 1.1),  # buffer on gripper DOF pos to prevent initial contact
    #                                       dim=-1)  # shape = (num_envs, num_dofs)
    #     self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
    #     self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

    #     multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
    #     self.gym.set_dof_state_tensor_indexed(self.sim,
    #                                           gymtorch.unwrap_tensor(self.dof_state),
    #                                           gymtorch.unwrap_tensor(multi_env_ids_int32),
    # 
    #                                          len(multi_env_ids_int32))
    
    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.cfg_task.randomize.randomize_compliance:
            if self.cfg_task.env.use_compliant_contact:
                # sample_compliance
                k_range = self.cfg_task.randomize.compliance_stiffness_range
                d_range = self.cfg_task.randomize.compliance_damping_range
                ks = k_range[0] + torch.rand(self.num_envs, device=self.device) * (k_range[1] - k_range[0])
                ds = d_range[0] + torch.rand(self.num_envs, device=self.device) * (d_range[1] - d_range[0])
                # set sampled compliance params for each env
                for elastomer_link_name in ['elastomer_left', 'elastomer_right']:
                    self.configure_compliant_dynamics(actor_handle=self.actor_handles['franka'],
                                                      elastomer_link_name=elastomer_link_name,
                                                      compliance_stiffness=ks,
                                                      compliant_damping=ds,
                                                      use_acceleration_spring=False)

        if self.cfg_task.randomize.randomize_ctrl_params:
            assert self.cfg_task.ctrl.ctrl_type == 'task_space_impedance', 'controller randomization currently works only for task_space_impedance'
            # use default controller params when randomizing initial state
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.task_space_impedance.task_deriv_gains, device=self.device).repeat(
                (self.num_envs, 1))

        if self.cfg_task.ige_dr.randomize:
            # use initial joint friction and damping during environment initialization
            for env_id in range(self.num_envs):
                env_ptr, franka_handle = self.env_ptrs[env_id], self.actor_handles['franka']
                franka_dof_props = self.gym.get_actor_dof_properties(env_ptr, franka_handle)
                franka_dof_props['friction'][7:9] = self.cfg_task.env.default_gripper_joint_friction
                franka_dof_props['damping'][7:9] = self.cfg_task.env.default_gripper_joint_damping
                self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)

        self._reset_franka(env_ids)
        self._reset_object(env_ids)

        self.disable_gravity()  # to prevent plug from falling
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)  # probably not needed
        self.refresh_all_tensors()

        self._move_gripper_to_dof_pos(gripper_dof_pos=self.cfg_task.env.get("franka_close_gripper_width", 0.0),
                                      sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
        self.enable_gravity(gravity_vec=self.cfg_base.sim.gravity)

        if self.cfg_task.randomize.randomize_ctrl_params:
            self.randomize_controller_params()

        if 'ige_dr' in self.cfg_task and self.cfg_task.ige_dr.randomize:
            # Must be executed before resetting self.reset_buf
            self.envs = self.env_ptrs  # DR code looks for self.envs
            self.apply_randomizations(self.cfg_task.ige_dr.randomization_params)
            self.set_gripper_friction_to_default()  # Don't randomize gripper friction/dynamic params, reset to default values

        self._reset_buffers(env_ids)

        unit_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        self.socket_obs_noise = unit_noise * torch.tensor(self.cfg_task.randomize.socket_pos_xyz_obs_noise,
                                                          device=self.device).expand(self.num_envs, 3)
        self.reset_image_augmentation()

    def set_gripper_friction_to_default(self):
        for env_id in range(self.num_envs):
            env_ptr, franka_handle = self.env_ptrs[env_id], self.actor_handles['franka']

            franka_dof_props = self.gym.get_actor_dof_properties(env_ptr, franka_handle)
            franka_dof_props['friction'][7:9] = self.cfg_task.env.default_gripper_joint_friction
            franka_dof_props['damping'][7:9] = self.cfg_task.env.default_gripper_joint_damping
            self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)

    def randomize_controller_params(self):

        k_gains_min = torch.tensor(self.cfg_task.randomize.task_prop_gains_min, dtype=torch.float32, device=self.device)
        k_gains_max = torch.tensor(self.cfg_task.randomize.task_prop_gains_max, dtype=torch.float32, device=self.device)
        ctrl_param_noise = torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device)

        self.cfg_ctrl['task_prop_gains'] = k_gains_min + ctrl_param_noise * (k_gains_max - k_gains_min)
        self.cfg_ctrl['task_deriv_gains'] = 2 * torch.sqrt(self.cfg_ctrl['task_prop_gains'])

        # Scale down the rotation gains of the controller. Stabilizes motion for the default franka urdf.
        self.cfg_ctrl['task_deriv_gains'][:, 3:] /= 10.     # reduce the damping of the rotation action params    

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        # shape of dof_pos = (num_envs, num_dofs)
        # shape of dof_vel = (num_envs, num_dofs)

        # Initialize Franka to initial joint configuration
        self.dof_pos[:, 0:7] = torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device)
        self.dof_pos[:, 7:] = self.cfg_task.env.get("franka_open_gripper_width",
                                                    self.asset_info_franka_table.franka_gripper_width_max)

        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.dof_vel[env_ids, 0:self.franka_num_dofs] = 0.0

        franka_actor_ids_sim_int32 = self.actor_ids_sim_tensors['franka'].to(dtype=torch.int32, device=self.device)[env_ids]
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(franka_actor_ids_sim_int32),
                                              len(franka_actor_ids_sim_int32))

        self._reset_franka_actuation(self.ctrl_target_dof_pos)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)  # probably not needed
        self.refresh_all_tensors()

        self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps,
                                     ctrl_target_gripper_dof_pos=self.cfg_task.env.get("franka_open_gripper_width",
                                                                                       self.asset_info_franka_table.franka_gripper_width_max))

    def _reset_franka_actuation(self, ctrl_target_dof_pos):
        multi_env_ids_int32 = self.actor_ids_sim_tensors['franka'].flatten()
        zeros = torch.zeros_like(self.dof_torque)
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(zeros),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(ctrl_target_dof_pos),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

    # def _reset_object(self, env_ids):
    #     """Reset root state of bulb."""

    #     # shape of root_pos = (num_envs, num_actors, 3)
    #     # shape of root_quat = (num_envs, num_actors, 4)
    #     # shape of root_linvel = (num_envs, num_actors, 3)
    #     # shape of root_angvel = (num_envs, num_actors, 3)

    #     bulb_pos = self.asset_info_franka_table.table_height + self.bulb_lengths[env_ids]
    #     self.root_pos[env_ids, self.bulb_actor_id_env] = \
    #         bulb_pos * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(len(env_ids), 1)
        
    #     bulb_rot = self.cfg_task.randomize.bulb_rot_initial * torch.ones((len(env_ids), 1), device=self.device) * math.pi / 180.0
    #     self.root_quat[env_ids, self.bulb_actor_id_env] = torch.cat((torch.zeros((len(env_ids), 1), device=self.device),
    #                                                                 torch.zeros((len(env_ids), 1), device=self.device),
    #                                                                 torch.sin(bulb_rot * 0.5),
    #                                                                 torch.cos(bulb_rot * 0.5)),
    #                                                                dim=-1)

    #     self.root_linvel[env_ids, self.bulb_actor_id_env] = 0.0
    #     self.root_angvel[env_ids, self.bulb_actor_id_env] = 0.0

    #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                  gymtorch.unwrap_tensor(self.root_state),
    #                                                  gymtorch.unwrap_tensor(self.bulb_actor_ids_sim),
    #                                                  len(self.bulb_actor_ids_sim))

    def _reset_object(self, env_ids):
        """Reset root states of bulb and socket."""
        self.gym.simulate(self.sim)
        self.refresh_base_tensors()

        # -----------------------------
        # Position bulb inside gripper
        # -----------------------------
        ee_to_bulb_tip_pos_local = torch.zeros_like(self.fingertip_midpoint_pos)

        # Ensure bulb can fully insert into socket
        min_tip_dist = self.socket_heights.squeeze(-1) * 1.1
        bulb_in_hand_buffer = self.cfg_task.env.bulb_in_hand_buffer
        min_tip_dist *= bulb_in_hand_buffer

        # Ensure bulb is still within gripper
        max_tip_dist = self.bulb_lengths.squeeze(-1) * 0.8

        # Compute range for tip distances
        tip_dist_range_center = (min_tip_dist + max_tip_dist) / 2.
        tip_dist_range_mag = max_tip_dist - min_tip_dist

        # Randomize bulb position in gripper
        bulb_pos_in_gripper_noise = 2 * (torch.rand((self.num_envs, 3), device=self.device) - 0.5)  # [-1, 1]
        bulb_pos_in_gripper_z_sampled = tip_dist_range_center + (
            bulb_pos_in_gripper_noise[:, 2] *
            tip_dist_range_mag / 2. *
            self.cfg_task.randomize.bulb_pos_z_in_gripper_noise_multiplier
        )

        # Subtract from bulb length → distance from tip
        ee_to_bulb_tip_pos_local[:, 2] = bulb_pos_in_gripper_z_sampled - self.bulb_lengths.squeeze(-1)

        # XY offsets in gripper
        bulb_pos_in_gripper_xy_sampled = bulb_pos_in_gripper_noise[:, :2] @ torch.diag(
            torch.tensor(self.cfg_task.randomize.bulb_pos_in_gripper_noise_xy, device=self.device)
        )
        ee_to_bulb_tip_pos_local[:, :2] = bulb_pos_in_gripper_xy_sampled

        # Compute world transform for bulb tip
        world_to_bulb_tip_quat, world_to_bulb_tip_pos = torch_jit_utils.tf_combine(
            self.fingertip_midpoint_quat,
            self.fingertip_midpoint_pos,
            self.identity_quat,
            ee_to_bulb_tip_pos_local
        )

        # -----------------------------
        # Add bulb rotation noise
        # -----------------------------
        bulb_noise_rot_in_gripper = 2 * (torch.rand((self.num_envs, 3), device=self.device) - 0.5)
        bulb_noise_rot_in_gripper *= torch.tensor(
            self.cfg_task.randomize.bulb_noise_rot_in_gripper, device=self.device
        ).expand(self.num_envs, 3)

        zero_translation = torch.zeros_like(world_to_bulb_tip_pos)
        ee_to_bulb_tip_rot_quat = torch_utils.quat_from_euler_xyz(
            bulb_noise_rot_in_gripper[:, 0],
            bulb_noise_rot_in_gripper[:, 1],
            bulb_noise_rot_in_gripper[:, 2]
        )

        world_to_bulb_tip_quat, world_to_bulb_tip_pos = torch_jit_utils.tf_combine(
            world_to_bulb_tip_quat,
            world_to_bulb_tip_pos,
            ee_to_bulb_tip_rot_quat,
            zero_translation
        )

        # -----------------------------
        # Translate from tip → base
        # -----------------------------
        bulb_tip_to_base_local = torch.zeros_like(self.fingertip_midpoint_pos)
        bulb_tip_to_base_local[:, 2] = self.bulb_lengths.squeeze()

        world_to_bulb_base_quat, world_to_bulb_base_pos = torch_jit_utils.tf_combine(
            world_to_bulb_tip_quat,
            world_to_bulb_tip_pos,
            self.identity_quat,
            bulb_tip_to_base_local
        )

        # Flip orientation (if needed)
        flip_z_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        world_to_bulb_base_quat, world_to_bulb_base_pos = torch_jit_utils.tf_combine(
            world_to_bulb_base_quat,
            world_to_bulb_base_pos,
            flip_z_quat,
            zero_translation
        )

        # Apply bulb root state
        self.root_pos[env_ids, self.bulb_actor_id_env, :] = world_to_bulb_base_pos
        self.root_quat[env_ids, self.bulb_actor_id_env] = world_to_bulb_base_quat
        self.root_linvel[env_ids, self.bulb_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.bulb_actor_id_env] = 0.0

        # -----------------------------
        # Randomize socket root state
        # -----------------------------
        socket_noise_xyz = 2 * (torch.rand((self.num_envs, 3), device=self.device) - 0.5)
        socket_noise_xyz = socket_noise_xyz @ torch.diag(
            torch.tensor(self.cfg_task.randomize.socket_pos_xyz_noise, device=self.device)
        )

        self.root_pos[env_ids, self.socket_actor_id_env, 0] = (
            self.cfg_task.randomize.socket_pos_xyz_initial[0] + socket_noise_xyz[env_ids, 0]
        )
        self.root_pos[env_ids, self.socket_actor_id_env, 1] = (
            self.cfg_task.randomize.socket_pos_xyz_initial[1] + socket_noise_xyz[env_ids, 1]
        )
        self.root_pos[env_ids, self.socket_actor_id_env, 2] = (
            self.cfg_task.randomize.socket_pos_xyz_initial[2] + socket_noise_xyz[env_ids, 2]
        )

        # Socket rotation (with noise)
        socket_rot_initial = self.cfg_task.randomize.socket_rot_initial
        socket_rot_noise_level = self.cfg_task.randomize.socket_rot_noise

        socket_rot_euler = torch.tensor(socket_rot_initial, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        socket_rot_noise = 2 * (torch.rand((self.num_envs, 3), device=self.device) - 0.5)
        socket_rot_noise = socket_rot_noise @ torch.diag(torch.tensor(socket_rot_noise_level, device=self.device))

        socket_rot_euler += socket_rot_noise
        self.root_quat[env_ids, self.socket_actor_id_env] = torch_utils.quat_from_euler_xyz(
            socket_rot_euler[:, 0], socket_rot_euler[:, 1], socket_rot_euler[:, 2]
        )

        self.root_linvel[env_ids, self.socket_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.socket_actor_id_env] = 0.0

        # -----------------------------
        # Push new states to sim
        # -----------------------------
        bulb_socket_actor_ids_sim = torch.cat((
            self.actor_ids_sim_tensors['bulb'][env_ids],
            self.actor_ids_sim_tensors['socket'][env_ids]
        ), dim=0)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(bulb_socket_actor_ids_sim),
            len(bulb_socket_actor_ids_sim)
        )


    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets or force/torque targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if self.cfg_task.rl.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
                                                                                                         1))
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if self.cfg_task.rl.unidirectional_force:
                force_actions[:, 2] = -(force_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _open_gripper(self, sim_steps=20):
        """Fully open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.1, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        # Keep current end-effector pose as target end-effector pose, when moving the gripper joint
        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_midpoint_pos.detach().clone()
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_midpoint_quat.detach().clone()

        self._move_to_target_pose_and_gripper_width(self.ctrl_target_fingertip_midpoint_pos,
                                                    self.ctrl_target_fingertip_midpoint_quat,
                                                    gripper_dof_pos, sim_steps=sim_steps)

    def _lift_gripper(self, gripper_dof_pos=0.0, lift_distance=0.3, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_midpoint_pos.detach().clone()
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_midpoint_quat.detach().clone()
        self.ctrl_target_fingertip_midpoint_pos[:, 2] += lift_distance
        self._move_to_target_pose_and_gripper_width(self.ctrl_target_fingertip_midpoint_pos,
                                                    self.ctrl_target_fingertip_midpoint_quat,
                                                    gripper_dof_pos, sim_steps=sim_steps)

    def _move_to_target_pose_and_gripper_width(self, target_fingertip_midpoint_pos, target_fingertip_midpoint_quat,
                                               gripper_dof_pos, sim_steps=20, gentle_gripper_close=False):
        """Move arm to target end-effector pose, and gripper to target width using task-space controller."""

        # Keep current end-effector pose as target end-effector pose, when moving the gripper joint
        self.ctrl_target_fingertip_midpoint_pos[:] = target_fingertip_midpoint_pos
        self.ctrl_target_fingertip_midpoint_quat[:] = target_fingertip_midpoint_quat

        # Step sim
        for step_i in range(sim_steps):
            self.refresh_all_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            if gentle_gripper_close:
                # break gripper motion into steps. Try to move to target in half the num of steps left.
                num_steps_left = sim_steps - step_i
                target_gripper_i = (gripper_dof_pos - self.gripper_dof_pos) / (0.5 * num_steps_left)
                target_gripper_pos = self.gripper_dof_pos + target_gripper_i
            else:
                target_gripper_pos = gripper_dof_pos
            self._apply_actions_as_ctrl_targets(delta_hand_pose, target_gripper_pos, do_scale=False)
            self.render()
            self.gym.simulate(self.sim)

    def _get_keypoint_dist(self, body):
        """Get keypoint distances."""

        axis_length = self.asset_info_franka_table.franka_hand_length + self.asset_info_franka_table.franka_finger_length

        if body == 'finger' or body == 'bulb':
            # Keypoint distance between finger/bulb and target
            if body == 'finger':
                self.keypoint1 = self.fingertip_midpoint_pos
                self.keypoint2 = fc.translate_along_local_z(pos=self.keypoint1,
                                                            quat=self.fingertip_midpoint_quat,
                                                            offset=-axis_length,
                                                            device=self.device)

            elif body == 'bulb':
                self.keypoint1 = self.bulb_com_pos
                self.keypoint2 = fc.translate_along_local_z(pos=self.bulb_com_pos,
                                                            quat=self.bulb_com_quat,
                                                            offset=axis_length,
                                                            device=self.device)

            self.keypoint1_targ = self.target_pos
            self.keypoint2_targ = self.keypoint1_targ + torch.tensor([0.0, 0.0, axis_length], device=self.device)

        elif body == 'finger_bulb':
            # Keypoint distance between finger and bulb
            self.keypoint1 = self.fingerpad_midpoint_pos
            self.keypoint2 = fc.translate_along_local_z(pos=self.keypoint1,
                                                        quat=self.fingertip_midpoint_quat,
                                                        offset=-axis_length,
                                                        device=self.device)

            self.keypoint1_targ = self.bulb_com_pos
            self.keypoint2_targ = fc.translate_along_local_z(pos=self.bulb_com_pos,
                                                             quat=self.bulb_com_quat,
                                                             offset=axis_length,
                                                             device=self.device)

        self.keypoint3 = self.keypoint1 + (self.keypoint2 - self.keypoint1) * 1.0 / 3.0
        self.keypoint4 = self.keypoint1 + (self.keypoint2 - self.keypoint1) * 2.0 / 3.0
        self.keypoint3_targ = self.keypoint1_targ + (self.keypoint2_targ - self.keypoint1_targ) * 1.0 / 3.0
        self.keypoint4_targ = self.keypoint1_targ + (self.keypoint2_targ - self.keypoint1_targ) * 2.0 / 3.0
        keypoint_dist = torch.norm(self.keypoint1_targ - self.keypoint1, p=2, dim=-1) \
                        + torch.norm(self.keypoint2_targ - self.keypoint2, p=2, dim=-1) \
                        + torch.norm(self.keypoint3_targ - self.keypoint3, p=2, dim=-1) \
                        + torch.norm(self.keypoint4_targ - self.keypoint4, p=2, dim=-1)

        return keypoint_dist

    # def _get_curr_successes(self):
    #     """Get success mask at current timestep."""

    #     curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    #     # If bulb is close enough to target pos
    #     is_close = torch.where(self.bulb_dist_to_target < self.thread_pitches.squeeze(-1),
    #                            torch.ones_like(curr_successes),
    #                            torch.zeros_like(curr_successes))

    #     curr_successes = torch.logical_or(curr_successes, is_close)

    #     return curr_successes

    def _get_curr_successes(self):
        """Get success mask at current timestep for bulb-socket insertion."""

        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # 1. Positional success: bulb tip is close enough to socket tip
        is_close = self.bulb_dist_to_target < self.cfg_task.rl.success_pos_thresh  # e.g. 0.01m

        # 2. Rotational success: bulb orientation aligns with socket orientation
        # compute quaternion distance or angle between bulb and socket
        quat_diff = torch_utils.quat_mul(self.bulb_quat, torch_utils.quat_conjugate(self.socket_quat))
        # angle between orientations
        angle_diff = 2 * torch.acos(torch.clamp(quat_diff[:, 3], -1.0, 1.0))
        is_aligned = angle_diff < self.cfg_task.rl.success_rot_thresh  # e.g. 10 degrees (0.174 rad)

        # Combine conditions: both must be true
        curr_successes = torch.logical_and(is_close, is_aligned)

        return curr_successes
    

    # def _get_curr_failures(self, curr_successes):
    #     """Get failure mask at current timestep."""

    #     curr_failures = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    #     # If max episode length has been reached
    #     self.is_expired = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length,
    #                                   torch.ones_like(curr_failures),
    #                                   curr_failures)

    #     # If bulb is too far from target pos
    #     self.is_far = torch.where(self.bulb_dist_to_target > self.cfg_task.rl.far_error_thresh,
    #                               torch.ones_like(curr_failures),
    #                               curr_failures)

    #     # If bulb has slipped (distance-based definition)
    #     self.is_slipped = \
    #         torch.where(
    #             self.bulb_dist_to_fingerpads > self.asset_info_franka_table.franka_fingerpad_length * 0.5 + self.bulb_heights.squeeze(-1) * 0.5,
    #             torch.ones_like(curr_failures),
    #             curr_failures)
    #     self.is_slipped = torch.logical_and(self.is_slipped, torch.logical_not(curr_successes))  # ignore slip if successful

    #     # If bulb has fallen (i.e., if bulb XY pos has drifted from center of bolt and bulb Z pos has drifted below top of bolt)
    #     self.is_fallen = torch.logical_and(
    #         torch.norm(self.bulb_com_pos[:, 0:2], p=2, dim=-1) > self.bolt_widths.squeeze(-1) * 0.5,
    #         self.bulb_com_pos[:, 2] < self.cfg_base.env.table_height + self.bolt_head_heights.squeeze(
    #             -1) + self.bolt_shank_lengths.squeeze(-1) + self.bulb_heights.squeeze(-1) * 0.5)

    #     curr_failures = torch.logical_or(curr_failures, self.is_expired)
    #     curr_failures = torch.logical_or(curr_failures, self.is_far)
    #     curr_failures = torch.logical_or(curr_failures, self.is_slipped)
    #     curr_failures = torch.logical_or(curr_failures, self.is_fallen)

    #     return curr_failures

    def _get_curr_failures(self, curr_successes):
        """Get failure mask at current timestep for bulb-socket task."""

        curr_failures = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.socket_radii = self.socket_diameters * 0.5

        # 1. Episode expired (timeout)
        self.is_expired = torch.where(
            self.progress_buf[:] >= self.cfg_task.rl.max_episode_length,
            torch.ones_like(curr_failures),
            curr_failures
        )

        # 2. Bulb is too far from socket target
        self.is_far = torch.where(
            self.bulb_dist_to_target > self.cfg_task.rl.far_error_thresh,
            torch.ones_like(curr_failures),
            curr_failures
        )

        # 3. Bulb slipped out of gripper (fingerpads cannot hold it)
        self.is_slipped = torch.where(
            self.bulb_dist_to_fingerpads >
            (self.asset_info_franka_table.franka_fingerpad_length * 0.6 +
            self.bulb_heights.squeeze(-1) * 0.5),
            torch.ones_like(curr_failures),
            curr_failures
        )
        # Ignore slip if already successful
        self.is_slipped = torch.logical_and(self.is_slipped, torch.logical_not(curr_successes))

        # 4. Bulb has fallen off the socket/table
        self.is_fallen = torch.logical_and(
            torch.norm(self.bulb_com_pos[:, 0:2] - self.socket_pos[:, 0:2], p=2, dim=-1) >
            self.socket_radii.squeeze(-1) * 0.5,  # drifted too far in XY plane
            self.bulb_com_pos[:, 2] <
            self.asset_info_franka_table.table_height + self.bulb_heights.squeeze(-1) * 0.5  # dropped below table height
        )

        # Combine all failure conditions
        curr_failures = torch.logical_or(curr_failures, self.is_expired)
        curr_failures = torch.logical_or(curr_failures, self.is_far)
        curr_failures = torch.logical_or(curr_failures, self.is_slipped)
        curr_failures = torch.logical_or(curr_failures, self.is_fallen)

        return curr_failures


    def _randomize_gripper_pose(self, env_ids, sim_steps, ctrl_target_gripper_dof_pos=0.0):
        """Move gripper to random pose."""

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_initial,
                                                               device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(
            self.num_envs, 1)

        fingertip_midpoint_pos_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                                                            device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        fingertip_midpoint_rot_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device))
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2])

        # Step sim and render
        for _ in range(sim_steps):
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(actions=actions,
                                                ctrl_target_gripper_dof_pos=ctrl_target_gripper_dof_pos,
                                                do_scale=False)

            self.gym.simulate(self.sim)
            self.refresh_all_tensors()
            self.render()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.actor_ids_sim_tensors['franka'][env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        self._reset_franka_actuation(self.dof_pos.clone())

        self.refresh_all_tensors()