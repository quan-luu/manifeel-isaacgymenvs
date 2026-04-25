import numpy as np
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
import time
import cv2
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Tuple


# Mappings from strings to environments
isaacgym_task_map = {
    "TacSLTaskInsertion": TacSLTaskInsertion,
    "TacSLTaskUSB": TacSLTaskUSB,
    "TacSLTaskGear": TacSLTaskGear,
    "TacSLTaskPowerInsertion": TacSLTaskPowerInsertion,
    "TacSLTaskPickInBox": TacSLTaskPickInBox,
    "TacSLTaskBoltNut": TacSLTaskBoltNut,
    "TacSLTaskBulb": TacSLTaskBulb, 
    "TacSLTaskSearchInBox": TacSLTaskSearchInBox,
    "TacSLTaskClassBall": TacSLTaskClassBall,
}

class ManifeelEnvWrapper():
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


    
if __name__ == '__main__':
            
    @hydra.main(version_base="1.1", 
                config_path="../isaacgymenvs/cfg", 
                config_name="isaacgym_config_bulb")
    def main(cfg: DictConfig):
        def visualize_tactile_shear_image(tactile_normal_force, tactile_shear_force,
                                          normal_force_threshold=0.0008, shear_force_threshold=0.001,
                                          resolution=24):
            """
            Visualize tactile normal/shear force as an arrow field image.

            Args:
                tactile_normal_force (np.ndarray): [H, W] normal force.
                tactile_shear_force (np.ndarray): [H, W, 2] shear force.
                normal_force_threshold (float): Threshold for normal-force color scaling.
                shear_force_threshold (float): Threshold for shear-force arrow scaling.
                resolution (int): Pixel size per tactile cell.

            Returns:
                np.ndarray: RGB image in uint8 for OpenCV display.
            """
            nrows = tactile_normal_force.shape[0]
            ncols = tactile_normal_force.shape[1]

            imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype=np.float32)

            for row in range(nrows):
                for col in range(ncols):
                    loc0_x = row * resolution + resolution // 2
                    loc0_y = col * resolution + resolution // 2
                    loc1_x = loc0_x + tactile_shear_force[row, col][0] / shear_force_threshold * resolution
                    loc1_y = loc0_y + tactile_shear_force[row, col][1] / shear_force_threshold * resolution
                    color = (
                        0.,
                        max(0., 1. - tactile_normal_force[row][col] / normal_force_threshold),
                        min(1., tactile_normal_force[row][col] / normal_force_threshold)
                    )

                    cv2.arrowedLine(
                        imgs_tactile,
                        (int(loc0_y), int(loc0_x)),
                        (int(loc1_y), int(loc1_x)),
                        color,
                        2,
                        tipLength=0.35
                    )

            imgs_tactile = np.clip(imgs_tactile * 255.0, 0.0, 255.0).astype(np.uint8)
            return imgs_tactile

        def summarize_array(name, value):
            if isinstance(value, torch.Tensor):
                array = value.detach().float().cpu()
                min_v = float(torch.min(array).item())
                max_v = float(torch.max(array).item())
                shape = tuple(array.shape)
            else:
                array = np.asarray(value)
                min_v = float(np.min(array))
                max_v = float(np.max(array))
                shape = array.shape
            print(f"{name}: shape={shape}, min={min_v:.6f}, max={max_v:.6f}")

        def tensor_to_bgr_image(obs_tensor: torch.Tensor) -> np.ndarray:
            image = obs_tensor.detach().cpu().numpy()
            if image.ndim == 4:
                image = image[0]

            if image.dtype != np.uint8:
                image = image.astype(np.float32)
                image_min = float(image.min())
                image_max = float(image.max())
                if image_min >= 0.0 and image_max <= 1.0:
                    image = image * 255.0
                image = np.clip(image, 0.0, 255.0).astype(np.uint8)

            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            return image

        # Pass the config explicitly
        wrapped_env = ManifeelEnvWrapper(cfg)
        print("Observation space is", wrapped_env.observation_space)
        print("Action space is", wrapped_env.action_space)
        wrapped_env.seed(0)
        obs = wrapped_env.reset()

        # NOTE: Set this variable to control how long the environment runs (in seconds).
        run_duration_sec = 10.0

        # Run the environment for approximately `run_duration_sec` seconds.
        start_time = time.monotonic()
        while (time.monotonic() - start_time) < run_duration_sec:
            random_actions = 2.0 * np.random.rand(wrapped_env.num_envs, wrapped_env.action_space.shape[0]) - 1.0
            # Uncomment the below line to set the last action dimension to a constant value (e.g., for gripper control).
            # random_actions[:, -1] = 0.020
            obs, reward, done, info = wrapped_env.step(random_actions)
            print("🚀obs_dict keys:", obs.keys())
            summarize_array("random_actions", random_actions)
            summarize_array("ee_pos", obs['ee_pos'])
            summarize_array("ee_quat", obs['ee_quat'])
            summarize_array("wrist", obs['wrist'])
            summarize_array("right_tactile_camera_taxim", obs['right_tactile_camera_taxim'])
            summarize_array("tactile_force_field_right", obs['tactile_force_field_right'])

            wrist_bgr = tensor_to_bgr_image(obs['wrist'])
            right_tactile_bgr = tensor_to_bgr_image(obs['right_tactile_camera_taxim'])
            tactile_force_field_right = obs['tactile_force_field_right'].detach().cpu().numpy()[0]
            right_shear_img = visualize_tactile_shear_image(
                tactile_force_field_right[..., 0],
                tactile_force_field_right[..., 1:]
            )
            cv2.imshow("wrist", wrist_bgr)
            cv2.imshow("right_tactile_camera_taxim", right_tactile_bgr)
            cv2.imshow("tactile_force_field_right_shear", right_shear_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stopping visualization loop because 'q' was pressed.")
                break
            
            # print(f"sucess: {wrapped_env.success}")
            # print(reward, done, info)

        cv2.destroyAllWindows()
            

    main()