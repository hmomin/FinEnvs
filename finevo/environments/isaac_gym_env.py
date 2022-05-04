import isaacgym
import torch
from .isaac_gym_envs.tasks import isaacgym_task_map
from .isaac_gym_envs.tasks.base.vec_task import VecTask
from .isaac_gym_envs.utils.config_utils import load_task_config
from typing import Dict, Tuple


class IsaacGymEnv:
    def __init__(
        self,
        env_name: str,
        num_envs: int = -1,
        headless: bool = False,
        sim_device_id: int = 0,
        graphics_device_id: int = 0,
    ):
        self.config = load_task_config(env_name)
        self.set_sim_and_graphics_devices(sim_device_id, graphics_device_id)
        isaac_task = isaacgym_task_map[env_name]
        self.override_default_num_envs(num_envs)
        self.env: VecTask = isaac_task(
            cfg=self.config,
            sim_device=self.sim_device,
            graphics_device_id=self.graphics_device_id,
            headless=headless,
        )

    def set_sim_and_graphics_devices(
        self, sim_device_id: int, graphics_device_id: int
    ) -> None:
        if torch.cuda.is_available():
            self.sim_device = f"cuda:{sim_device_id}"
            self.graphics_device_id = graphics_device_id
        else:
            print("WARNING: PyTorch not recognizing CUDA device -> forcing CPU...")
            self.sim_device = "cpu"
            self.graphics_device_id = -1

    def override_default_num_envs(self, num_envs: int) -> None:
        if num_envs > 0:
            self.config["env"]["numEnvs"] = num_envs
            self.num_envs = num_envs
        else:
            self.num_envs = self.config["env"]["numEnvs"]

    def reset(self) -> torch.Tensor:
        states = self.env.reset()
        if isinstance(states, Dict):
            states = states["obs"]
        assert isinstance(states, torch.Tensor)
        return states

    def reset_all(self) -> torch.Tensor:
        env_indices = torch.arange(0, self.num_envs, device=self.sim_device).long()
        self.env.reset_idx(env_indices)
        return self.reset()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        (observations_dict, rewards, dones, info_dict) = self.env.step(actions)
        observations = observations_dict["obs"]
        return (observations, rewards, dones, info_dict)
