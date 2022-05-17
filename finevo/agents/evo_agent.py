import csv
import os
import torch
from datetime import datetime
from ..base_object import BaseObject
from ..device_utils import set_device
from .networks.parallel_mlp import ParallelMLP
from time import time
from typing import Tuple, Dict


class EvoAgent(BaseObject):
    def __init__(
        self,
        env_args: Dict,
        hidden_dims: Tuple[int] = (128, 128),
        learning_rate: float = 0.01,
        noise_std_dev: float = 0.02,
        l2_coefficient: float = 0.0,
        write_to_csv: bool = True,
        device_id: int = 0,
    ):
        self.set_env_params(env_args)
        self.device = set_device(device_id)
        self.learning_rate = learning_rate
        self.noise_std_dev = noise_std_dev
        self.network_shape = (self.num_observations, *hidden_dims, self.num_actions)
        self.network = ParallelMLP(
            self.num_envs,
            self.network_shape,
            learning_rate=learning_rate,
            noise_std_dev=noise_std_dev,
            l2_coefficient=l2_coefficient,
            device_id=device_id,
        )
        self.network.perturb_parameters()
        self.returns = torch.zeros(
            (self.num_envs,), device=self.device, requires_grad=False
        )
        self.dones = torch.zeros(
            (self.num_envs,), device=self.device, requires_grad=False
        )
        self.write_to_csv = write_to_csv
        if write_to_csv:
            self.create_progress_log()

    def set_env_params(self, env_args: Dict) -> None:
        self.env_name: str = env_args["env_name"]
        self.num_envs: int = env_args["num_envs"]
        self.num_observations: int = env_args["num_observations"]
        self.num_actions: int = env_args["num_actions"]

    def create_progress_log(self) -> None:
        trials_dir = os.path.join(os.getcwd(), "trials")
        if not os.path.exists(trials_dir):
            os.mkdir(trials_dir)
        self.csv_name = os.path.join(
            trials_dir,
            datetime.now().strftime(f"{self.env_name}_Evo_%Y-%m-%d_%H-%M-%S.csv"),
        )
        csv_fields = [
            "unix_time",
            "num_episodes",
            "eval_return",
            "max_return",
            "mean_return",
            "std_dev_return",
        ]
        with open(self.csv_name, "a") as f:
            writer = csv.writer(f)
            writer.writerow(csv_fields)

    def step(self, states: torch.Tensor) -> torch.Tensor:
        actions = self.network.forward(states)
        return actions

    def store(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> int:
        current_done_indices = (self.dones == 1).nonzero()
        new_done_indices = (dones == 1).nonzero()
        rewards[current_done_indices] = 0
        self.returns += rewards
        self.dones[new_done_indices] = 1
        return sum(self.dones)

    def log_progress(self) -> float:
        num_episodes = self.returns.shape[0]
        eval_return = self.returns[-1].item()
        max_return = self.returns.max().item()
        mean_return = self.returns.mean().item()
        std_dev_return = self.returns.std().item()
        record_fields = [
            time(),
            num_episodes,
            eval_return,
            max_return,
            mean_return,
            std_dev_return,
        ]
        if self.write_to_csv:
            with open(self.csv_name, "a") as f:
                writer = csv.writer(f)
                writer.writerow(record_fields)
        print(
            f"eval return: {eval_return:.6f} | "
            + f"max return: {max_return:.6f} | "
            + f"mean return: {mean_return:.6f} | "
            + f"std dev return: {std_dev_return:.6f}"
        )
        self.returns = torch.zeros((self.num_envs,), device=self.device)
        self.dones = torch.zeros((self.num_envs,), device=self.device)
        return eval_return

    def train(self) -> float:
        self.network.reconstruct_perturbations()
        self.network.update_parameters(self.returns)
        eval_return = self.log_progress()
        self.network.perturb_parameters()
        return eval_return
