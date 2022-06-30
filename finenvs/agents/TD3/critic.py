import torch
import torch.nn as nn
from ..networks.generic_network import GenericNetwork
from ..networks.lstm import LSTMNetwork
from ..networks.multilayer_perceptron import MLPNetwork


class Critic(GenericNetwork):
    def __init__(self, device_id: int = 0):
        super().__init__(device_id=device_id)

    def __new__(cls, *args, **kwargs):
        if cls is Critic:
            raise TypeError(
                f"'{cls.__name__}' should not be directly instantiated. "
                + f"Try '{cls.__name__}MLP' or '{cls.__name__}LSTM' instead."
            )
        return object.__new__(cls)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
        retain_graph=False,
    ) -> None:
        loss = self.compute_loss(states, actions, targets)
        self.gradient_descent_step(loss, retain_graph)

    def compute_loss(
        self, states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat([states.float(), actions], dim=1)
        state_action_values: torch.Tensor = self.forward(inputs)
        mean_square_error = (state_action_values - targets).square().mean()
        return mean_square_error

    def gradient_descent_step(
        self, loss: torch.Tensor, retain_graph: bool = False
    ) -> None:
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()


class CriticMLP(MLPNetwork, Critic):
    def __init__(
        self,
        shape: tuple,
        learning_rate: float = 3e-4,
        layer_activation=nn.ELU,
        output_activation=nn.Identity,
        device_id: int = 0,
    ):
        super().__init__(
            shape=shape,
            layer_activation=layer_activation,
            output_activation=output_activation,
            device_id=device_id,
        )
        self.create_optimizer(learning_rate)


class CriticLSTM(LSTMNetwork, Critic):
    def __init__(
        self,
        shape: tuple,
        learning_rate: float = 3e-4,
        output_activation=nn.Identity,
        device_id: int = 0,
    ):
        super().__init__(
            shape=shape,
            output_activation=output_activation,
            device_id=device_id,
        )
        self.create_optimizer(learning_rate)
