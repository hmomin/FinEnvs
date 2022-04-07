import torch
import torch.nn as nn
from .networks.generic_network import GenericNetwork
from .networks.lstm import LSTMNetwork
from .networks.multilayer_perceptron import MLPNetwork


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

    def gradient_descent_step(self, states: torch.Tensor, returns: torch.Tensor):
        self.optimizer.zero_grad()
        loss = self.compute_critic_loss(states, returns)
        loss.backward()
        self.optimizer.step()

    def compute_critic_loss(
        self, states: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        values = self.forward(states)
        squared_errors = (returns - values) ** 2
        mean_squared_error = squared_errors.mean()
        return mean_squared_error


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
