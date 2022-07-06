import torch
from typing import Dict
from ..base_object import BaseObject
from ..device_utils import set_device


class Buffer(BaseObject):
    def __init__(self, max_size: int = 1_000_000, device_id: int = 0) -> None:
        self.max_size = max_size
        self.device = set_device(device_id)
        self.container: Dict[str, torch.Tensor] = {
            "states": None,
            "actions": None,
            "rewards": None,
            "next_states": None,
            "dones": None,
        }

    def force_2D(self, tensor: torch.Tensor) -> torch.Tensor:
        while len(tensor.shape) < 2:
            tensor = tensor.unsqueeze(-1)
        return tensor

    def store_tensor(self, key: str, tensor: torch.Tensor) -> None:
        tensor = self.force_2D(tensor)
        if self.container[key] == None:
            self.container[key] = tensor.float()
        else:
            self.container[key] = torch.cat(
                [self.container[key], tensor], dim=0
            ).float()

    def store(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.store_tensor("states", states)
        self.store_tensor("actions", actions)
        self.store_tensor("rewards", rewards)
        self.store_tensor("next_states", next_states)
        self.store_tensor("dones", dones)
        if self.size() > self.max_size:
            self.discard_old_data()

    def size(self) -> int:
        dones = self.container["dones"]
        if dones == None:
            return 0
        else:
            return dones.shape[0]

    def discard_old_data(self):
        current_size = self.size()
        assert (
            current_size > self.max_size
        ), "Attempted to discard old values from buffer that's not at max capacity."
        excess = current_size - self.max_size
        new_indices = torch.arange(excess, current_size, 1, device=self.device)
        for key in self.container.keys():
            self.container[key] = torch.index_select(
                self.container[key], 0, new_indices
            )

    def get_mini_batch(self, size: int) -> Dict[str, torch.Tensor]:
        indices = torch.randint(0, self.size(), (size,), device=self.device)
        keys = self.container.keys()
        values = self.container.values()
        return dict(
            [
                (key, torch.index_select(value, 0, indices))
                for key, value in zip(keys, values)
            ]
        )
