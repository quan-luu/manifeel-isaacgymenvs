import numpy as np
# numpy compatibility fix for older versions (e.g., in TVB)
def patch_numpy_compatibility():
    if not hasattr(np, "bool"):
        np.bool = bool
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "object"):
        np.object = object

patch_numpy_compatibility() # Apply the patch before importing any modules that might use numpy

from gym import spaces

import isaacgym
import isaacgymenvs

from isaacgymenvs.tasks.tacsl.tacsl_task_bulb import TacSLTaskBulb
from isaacgymenvs.utils.utils import set_seed
from isaacgymenvs.tasks.tacsl.tacsl_task_insertion import TacSLTaskInsertion
from isaacgymenvs.tasks.tacsl.tacsl_task_gear import TacSLTaskGear
from isaacgymenvs.tasks.tacsl.tacsl_task_USB import TacSLTaskUSB
from isaacgymenvs.tasks.tacsl.tacsl_task_power import TacSLTaskPowerInsertion
from isaacgymenvs.tasks.tacsl.tacsl_task_bolt_nut import TacSLTaskBoltNut
from isaacgymenvs.tasks.tacsl.tacsl_task_pick_in_box import TacSLTaskPickInBox
from isaacgymenvs.tasks.tacsl.tacsl_task_search_in_box import TacSLTaskSearchInBox
from isaacgymenvs.tasks.tacsl.tacsl_task_class_ball import TacSLTaskClassBall
from isaacgymenvs.utils.reformat import omegaconf_to_dict

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Tuple
from furniture_bench.robot.robot_state import ROBOT_STATE_DIMS, ROBOT_STATES

# Mappings from strings to environments
isaacgym_task_map = {
    "TacSLTaskInsertion": TacSLTaskInsertion,
    "TacSLTaskUSB": TacSLTaskUSB,
    "TacSLTaskGear": TacSLTaskGear,
    "TacSLTaskPowerInsertion": TacSLTaskPowerInsertion,
    "TacSLTaskPickInBox": TacSLTaskPickInBox,
    "TacSLTaskBoltNut": TacSLTaskBoltNut,
    # "TacSLTaskBulbSocketScrew": TacSLTaskBulbSocketScrew,
    "TacSLTaskBulb": TacSLTaskBulb, # alias for backward compatibility
    "TacSLTaskSearchInBox": TacSLTaskSearchInBox,
    "TacSLTaskClassBall": TacSLTaskClassBall,
}

class IsaacEnvWrapper():
    def __init__(self, cfg):
        self.cfg = cfg
        self._start_task(self.cfg)
        self.n_parts_assemble = 1

    @property
    def observation_space(self):
        return self.envs.observation_space
    
    @property
    def action_space(self):
        return self.envs.action_space

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.envs.num_actions

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.envs.num_observations

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.envs.num_environments

    @property
    def success(self):
        return self.envs._check_success()

    @property
    def device(self):
        return self.envs.device
    
    def _start_task(self, cfg):
        rl_device = cfg.rl_device
        sim_device = cfg.sim_device
        graphics_device_id = cfg.graphics_device_id
        headless = cfg.headless
        virtual_screen_capture = cfg.capture_video
        force_render = cfg.force_render

        cfg_task = cfg['task']
        cfg_dict = omegaconf_to_dict(cfg_task)

        task_name = cfg_dict['name'] #e.g., TacSLTaskUSB

        self.envs = isaacgym_task_map[task_name](cfg=cfg_dict,
                                       rl_device=rl_device,
                                       sim_device=sim_device,
                                       graphics_device_id=graphics_device_id,
                                       headless=headless,
                                       virtual_screen_capture=virtual_screen_capture,
                                       force_render=force_render,
                                       )

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        # global rank of the GPU
        # global_rank = int(os.getenv("RANK", "0"))
        global_rank = 0
        # sets seed. if seed is -1 will pick a random one
        seed = set_seed(seed, torch_deterministic=self.cfg.torch_deterministic, rank=global_rank)
        # seed = set_seed(seed)
        self._seed = seed

    def reset(self):
        # obs = self.envs.reset()
        self.envs.reset_idx(torch.arange(self.num_envs))
        self.envs.compute_observations()
        obs = self.envs.reset()
        obs = obs['obs']
        # print("🚀obs_dict keys:", obs.keys())
        return obs

    def step(self, action):
        # actions = torch.from_numpy(action).to(dtype=torch.float32).unsqueeze(0)
        if isinstance(action, torch.Tensor):
            actions = action.to(dtype=torch.float32)
        else:
            actions = torch.from_numpy(action).to(dtype=torch.float32)

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        
        obs, reward, reset, info = self.envs.step(actions)
        obs = obs['obs']

        return obs, reward, reset, info

    
    def render(self, mode="rgb_array"):
        if self.envs.virtual_display and mode == "rgb_array":
            img = self.envs.virtual_display.grab()
            return np.array(img)
        


    def close(self):
        pass

    # -------------------------------------------------
    # ENV STATE SNAPSHOT (Deterministic Undo / Resume)
    # -------------------------------------------------

    def get_env_state(self, env_idx: int = 0) -> dict:
        """
        Snapshot full simulation state for one environment index.

        Works for vectorized IsaacGym envs.
        """

        # Make sure tensors are fresh
        # self.envs.gym.refresh_actor_root_state_tensor(self.envs.sim)
        # self.envs.gym.refresh_dof_state_tensor(self.envs.sim)
        # self.envs.gym.refresh_rigid_body_state_tensor(self.envs.sim)
        self.envs.refresh_base_tensors()

        # Clone tensors (IMPORTANT: clone, not reference)
        root_states = self.envs.root_state.clone()
        dof_states = self.envs.dof_state.clone()
        rb_states = self.envs.body_state.clone()

        state = {
            "root_state_tensor": root_states,
            "dof_state_tensor": dof_states,
            "rigid_body_state_tensor": rb_states,
        }

        # Optional buffers (only if exist)
        if hasattr(self, "progress_buf"):
            state["progress_buf"] = self.envs.progress_buf.clone()

        if hasattr(self, "reset_buf"):
            state["reset_buf"] = self.envs.reset_buf.clone()

        if hasattr(self, "success"):
            state["success"] = self.envs._check_success().clone()

        if hasattr(self, "episode_length_buf"):
            state["episode_length_buf"] = self.envs.episode_length_buf.clone()

        # Save RNG states for full determinism
        state["torch_rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()

        return state


    def set_env_state(self, env_idx: int = 0, state: dict = None):
        """
        Restore full simulation state.

        Must be followed by refresh().
        """

        if state is None:
            return

        # Restore simulation tensors
        self.envs.root_state.copy_(state["root_state_tensor"])
        self.envs.dof_state.copy_(state["dof_state_tensor"])
        self.envs.body_state.copy_(state["rigid_body_state_tensor"])

        # Push tensors back into simulator
        self.envs.gym.set_actor_root_state_tensor(self.envs.sim, self.envs.root_state)
        self.envs.gym.set_dof_state_tensor(self.envs.sim, self.envs.dof_state)

        # Restore buffers if present
        if "progress_buf" in state and hasattr(self, "progress_buf"):
            self.envs.progress_buf.copy_(state["progress_buf"])

        if "reset_buf" in state and hasattr(self, "reset_buf"):
            self.envs.reset_buf.copy_(state["reset_buf"])

        if "success_buf" in state and hasattr(self, "success"):
            self.envs.success.copy_(state["success_buf"])

        if "episode_length_buf" in state and hasattr(self, "episode_length_buf"):
            self.envs.episode_length_buf.copy_(state["episode_length_buf"])

        # Restore RNG states
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in state:
            torch.cuda.set_rng_state(state["cuda_rng_state"])


    def refresh(self):
        """
        Refresh all simulation tensors after restoring state.
        """

        # self.envs.gym.refresh_actor_root_state_tensor(self.envs.sim)
        # self.envs.gym.refresh_dof_state_tensor(self.envs.sim)
        # self.envs.gym.refresh_rigid_body_state_tensor(self.envs.sim)
        self.envs.refresh_base_tensors()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output
    import torch
            
    @hydra.main(version_base="1.1", 
                config_path="../../TVB/TVB/config", 
                config_name="isaacgym_config")
    def main(cfg: DictConfig):
        # Pass the config explicitly
        wrapped_env = IsaacEnvWrapper(cfg)
        print("Observation space is", wrapped_env.observation_space)
        print("Action space is", wrapped_env.action_space)
        wrapped_env.seed(0)
        obs = wrapped_env.reset()
        for _ in range(5):
            random_actions = 2.0 * np.random.rand(wrapped_env.action_space.shape[0]) - 1.0
            obs, reward, done, info = wrapped_env.step(random_actions)
            print("🚀obs_dict keys:", obs.keys())
            print(f"🚀robot_state.shape: {obs['robot_state'].shape}")
            print(f"🚀color_image1.shape: {obs['color_image1'].shape}")
            print(f"🚀color_image2.shape: {obs['color_image2'].shape}")
            print(f"sucess: {wrapped_env.success}")
            # print(f"🚀obs_dict[tactile_force_field_left].shape: {obs['tactile_force_field_left'].shape}")
            # print(f"🚀obs_dict[tactile_depth_left].shape: {obs['tactile_depth_left'].shape}")

            print(reward, done, info)
            # tactile_rgb_image = observations['left_tactile_camera_taxim'][0]
            # wrist_rgb_image = observations['wrist_2'][0]

    main()