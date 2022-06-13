import numpy as np
import torch
from ...base_object import BaseObject
from ...device_utils import set_device
from typing import Dict


class Buffer(BaseObject):
    def __init__(
        self, num_mini_batches: int = 16, gamma: float = 0.99, device_id: int = 0
    ):
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.device = set_device(device_id)
        self.container: Dict[str, torch.Tensor] = {
            "states": None,
            "actions": None,
            "rewards": None,
            "dones": None,
            "log_probs": None,
            "values": None,
            "advantages": None,
            "returns": None,
        }
        self.batch_keys = [
            "states",
            "actions",
            "log_probs",
            "advantages",
            "returns",
        ]

    def store(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self.store_tensor("states", states)
        self.store_tensor("actions", actions)
        self.store_tensor("rewards", rewards)
        self.store_tensor("dones", dones)
        self.store_tensor("log_probs", log_probs)
        self.store_tensor("values", values)

    def store_tensor(self, key: str, tensor: torch.Tensor) -> None:
        tensor = self.force_3D(tensor)
        if self.container[key] == None:
            self.container[key] = tensor
        else:
            self.container[key] = torch.cat([self.container[key], tensor], dim=1)

    def force_3D(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) < 2:
            tensor = torch.unsqueeze(tensor, -1)
        if len(tensor.shape) < 3:
            tensor = torch.unsqueeze(tensor, 1)
        return tensor

    def size(self) -> int:
        states = self.container["states"]
        if states == None:
            return 0
        elif len(states.shape) == 3:
            (num_envs, num_steps, num_obs) = states.shape
            return num_envs * num_steps
        else:
            (total_steps, num_obs) = states.shape
            return total_steps

    def prepare_training_data(self, current_state_values: torch.Tensor) -> None:
        self.compute_returns_and_advantages(current_state_values)
        self.reshape()

    def compute_returns_and_advantages(self, last_values: torch.Tensor) -> None:
        gamma = self.gamma
        rewards = self.container["rewards"]
        dones = self.container["dones"]
        values = self.container["values"]
        (num_envs, num_steps, _) = rewards.shape
        with torch.no_grad():
            returns = torch.zeros(
                (num_envs, num_steps, _), device=self.device, requires_grad=False
            )
            current_returns = last_values
            for idx in reversed(range(num_steps)):
                current_rewards = rewards[:, idx, :]
                current_dones = dones[:, idx, :]
                current_returns = (
                    current_rewards
                    + (1 - current_dones) * gamma * current_returns.clone()
                )
                returns[:, idx, :] = current_returns
            self.container["returns"] = returns
            self.container["advantages"] = returns - values

    def reshape(self) -> None:
        for key, tensor in self.container.items():
            (num_envs, num_steps, last_dim) = tensor.shape
            self.container[key] = torch.reshape(
                tensor, (num_envs * num_steps, last_dim)
            )

    def get_batches(self) -> Dict[str, torch.Tensor]:
        self.shuffle()
        batch_dict = {}
        for key in self.batch_keys:
            batch_dict[key] = self.container[key]
        return batch_dict

    def shuffle(self) -> None:
        with torch.no_grad():
            random_indices = torch.randperm(
                self.size(), device=self.device, requires_grad=False
            )
            for key, tensor in self.container.items():
                permuted_tensor = tensor[random_indices, :]
                self.container[key] = permuted_tensor

    def get_mini_batch_indices(self) -> "list[torch.Tensor]":
        buffer_size = self.size()
        num_mini_batches = self.num_mini_batches
        mini_batch_size = int(np.floor(buffer_size / num_mini_batches))
        if mini_batch_size * num_mini_batches != buffer_size:
            print(
                f"WARNING: buffer size {buffer_size} does not divide evenly into {num_mini_batches} mini-batches!"
            )
        mini_batch_indices = []
        for mini_batch_idx in range(num_mini_batches):
            start_idx = mini_batch_idx * mini_batch_size
            end_idx = (mini_batch_idx + 1) * mini_batch_size - 1
            current_mini_batch_indices = torch.linspace(
                start_idx,
                end_idx,
                mini_batch_size,
                dtype=torch.long,
                device=self.device,
                requires_grad=False,
            )
            mini_batch_indices.append(current_mini_batch_indices)
        return mini_batch_indices

    def clear(self) -> None:
        for key in self.container.keys():
            self.container[key] = None
