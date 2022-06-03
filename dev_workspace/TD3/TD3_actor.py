import numpy as np
import torch
import torch.nn as nn
from finevo.agents.networks.generic_network import GenericNetwork
from finevo.agents.networks.multilayer_perceptron import MLPNetwork
from finevo.agents.networks.lstm import LSTMNetwork
from torch.distributions.normal import Normal


class TD3Actor(GenericNetwork):
    def __init__(self, device_id: int = 0):
        super().__init__(device_id=device_id)

    def __new__(cls, *args, **kwargs):
        if cls is TD3Actor:
            raise TypeError(
                f"'{cls.__name__}' should not be directly initiated. "
                + f"Try '{cls.__name__}MLP or {cls.__name__}LSTM' instead"
            )
        return super().__new__(cls)

    def set_up_parameters(
        self,
        shape: tuple,
        learning_rate: float,
        starting_std_dev: float,
        clip_epsilon: float,
    ):
        self.clip_epsilon = clip_epsilon
        num_actions = shape[-1]

        self.starting_std_dev = nn.Parameter(
            torch.ones((1, num_actions), device=self.device) * starting_std_dev
        )

        self.mean = nn.Parameter(torch.ones((1, num_actions), device=self.device))

        self.create_optimizer(learning_rate=learning_rate)

    def take_noised_action(self, inputs: torch.Tensor) -> torch.Tensor:
        action = self.forward(inputs)
        noise = Normal(self.mean, self.starting_std_dev)
        noise = noise.sample()
        noised_action = torch.clamp(action + noise, action.min(), action.max())
        return noised_action

    def gradient_ascent_step(
        self,
        state_action_values: torch.Tensor,
    ) -> None:
        loss = -state_action_values
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class TD3ActorMLP(MLPNetwork, TD3Actor):
    def __init__(
        self,
        shape: tuple,
        learning_rate: float = 3e-4,
        starting_std_dev: float = 1.0,
        clip_epsilon: float = 0.2,
        layer_activation=nn.ELU,
        output_activation=nn.Tanh,
        device_id: int = 0,
    ):
        super().__init__(
            shape,
            layer_activation=layer_activation,
            output_activation=output_activation,
            device_id=device_id,
        )
        self.set_up_parameters(shape, learning_rate, starting_std_dev, clip_epsilon)


class TD3ActorLSTM(LSTMNetwork, TD3Actor):
    def __init__(
        self,
        shape: tuple,
        learning_rate: float = 3e-4,
        starting_std_dev: float = 1.0,
        clip_epsilon: float = 0.2,
        output_activation=nn.Tanh,
        device_id: int = 0,
    ):
        super().__init__(
            shape, output_activation=output_activation, device_id=device_id
        )
        self.set_up_parameters(shape, learning_rate, starting_std_dev, clip_epsilon)
