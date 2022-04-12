import torch
import torch.nn as nn
from .generic_network import GenericNetwork


class MLPNetwork(GenericNetwork):
    def __init__(
        self,
        shape: tuple,
        layer_activation=nn.ELU,
        output_activation=nn.Identity,
        device_id: int = 0,
    ):
        super().__init__(device_id=device_id)
        self.create_layers(shape, layer_activation, output_activation)

    def create_layers(self, shape: tuple, layer_activation, output_activation) -> None:
        layers = []
        for idx in range(len(shape) - 1):
            activation = layer_activation if idx < len(shape) - 2 else output_activation
            layers.append(nn.Linear(shape[idx], shape[idx + 1], device=self.device))
            layers.append(activation())
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
