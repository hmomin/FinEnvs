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
        self.current_returns = torch.zeros((self.num_envs,), device=self.device)
        self.finished_returns = torch.zeros(0, device=self.device)
        self.dones = torch.zeros(0, dtype=torch.long, device=self.device)
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
            "L2_norm",
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
        self.current_returns += rewards
        new_done_indices = torch.squeeze(dones.nonzero(), 1)
        finished_returns = self.current_returns[new_done_indices]
        self.dones = torch.cat([self.dones, new_done_indices], dim=0)
        self.finished_returns = torch.cat(
            [self.finished_returns, finished_returns], dim=0
        )
        self.current_returns[new_done_indices] = 0
        return self.finished_returns.nelement()

    def log_progress(self) -> float:
        self.compute_mean_returns()
        num_episodes = self.mean_returns.shape[0]
        eval_return = self.mean_returns[-1].item()
        max_return = self.mean_returns.max().item()
        mean_return = self.mean_returns.mean().item()
        std_dev_return = self.mean_returns.std().item()
        L2_norm = self.network.get_l2_norm()
        record_fields = [
            time(),
            num_episodes,
            eval_return,
            max_return,
            mean_return,
            std_dev_return,
            L2_norm,
        ]
        if self.write_to_csv:
            with open(self.csv_name, "a") as f:
                writer = csv.writer(f)
                writer.writerow(record_fields)
        print(
            f"eval return: {eval_return:.6f} | "
            + f"max return: {max_return:.6f} | "
            + f"mean return: {mean_return:.6f} | "
            + f"std dev return: {std_dev_return:.6f} | "
            + f"L2 norm: {L2_norm:.6f}"
        )
        del self.centered_ranks, self.final_ranks, self.mean_returns
        self.current_returns = torch.zeros((self.num_envs,), device=self.device)
        self.finished_returns = torch.zeros(0, device=self.device)
        self.dones = torch.zeros(0, dtype=torch.long, device=self.device)
        return eval_return

    def compute_mean_returns(self) -> None:
        done_counts = torch.bincount(self.dones, minlength=self.num_envs)
        numerator = torch.zeros((self.num_envs,), device=self.device)
        numerator.index_add_(0, self.dones, self.finished_returns)
        denominator = done_counts.float()
        # avoid dividing by zero
        denominator[done_counts == 0] = 1.0
        self.mean_returns = numerator / denominator

    def train(self) -> float:
        self.network.reconstruct_perturbations()
        self.perform_rank_transformation()
        self.network.update_parameters(self.final_ranks)
        eval_return = self.log_progress()
        self.network.perturb_parameters()
        # self.network.perturb_parameters_2()
        return eval_return

    def perform_rank_transformation(self) -> None:
        sort_indices = self.finished_returns.argsort()
        self.compute_centered_ranks(sort_indices)
        self.compute_final_ranks()

    def compute_centered_ranks(self, sort_indices: torch.Tensor) -> None:
        ranks = torch.empty(sort_indices.shape, device=self.device)
        linear_ranks = torch.arange(
            0, sort_indices.shape[0], dtype=torch.float32, device=self.device
        )
        ranks[sort_indices] = linear_ranks
        N = len(ranks)
        self.centered_ranks = ranks / (N - 1) - 0.5

    def compute_final_ranks(self) -> None:
        self.final_ranks = torch.zeros((self.num_envs,), device=self.device)
        self.final_ranks.index_add_(0, self.dones, self.centered_ranks)
