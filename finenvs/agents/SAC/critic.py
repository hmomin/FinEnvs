import torch
import torch.nn as nn
from ..networks.generic_network import GenericNetwork
from ..networks.lstm import LSTMNetwork
from ..networks.multilayer_perceptron import MLPNetwork


class Critic(GenericNetwork):
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)

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
    ) -> None:
        loss = self.compute_loss(states, actions, targets)
        self.gradient_descent_step(loss)

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat([states, actions], dim=1)
        state_action_values: torch.Tensor = self.forward(inputs)
        mean_square_error = (state_action_values - targets).square().mean()
        return mean_square_error

    def gradient_descent_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class CriticMLP(MLPNetwork, Critic):
    def __init__(
        self,
        shape: tuple,
        learning_rate=3e-4,
        layer_activation=nn.ELU,
        output_activation=nn.Identity,
        device_id: int = 0,
    ):
        super().__init__(shape, layer_activation, output_activation, device_id)
        self.create_optimizer(learning_rate)


class CriticLSTM(LSTMNetwork, Critic):
    def __init__(
        self,
        shape: tuple,
        sequence_length: int,
        learning_rate=3e-4,
        output_activation=nn.Identity,
        device_id: int = 0,
    ):
        super().__init__(shape, sequence_length, output_activation, device_id)
        self.create_optimizer(learning_rate)
