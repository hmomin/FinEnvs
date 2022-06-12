import torch
import torch.nn as nn
from .TD3_Critic import TD3Critic
from finevo.agents.networks.generic_network import GenericNetwork
from finevo.agents.networks.multilayer_perceptron import MLPNetwork
from finevo.agents.networks.lstm import LSTMNetwork


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
        standard_deviation: float,
    ):
        self.shape = shape
        self.standard_deviation = standard_deviation
        self.create_optimizer(learning_rate=learning_rate)

    def get_noisy_actions(self, inputs: torch.Tensor) -> torch.Tensor:
        deterministic_actions = self.get_deterministic_actions(inputs)
        noise = (
            torch.randn(*deterministic_actions.shape, device=self.device)
            * self.standard_deviation
        )
        noisy_actions = deterministic_actions + noise
        clamped_noisy_actions = torch.clamp(noisy_actions, -1, +1)
        return clamped_noisy_actions

    def get_deterministic_actions(self, inputs: torch.Tensor) -> torch.Tensor:
        actions: torch.Tensor = self.forward(inputs).detach()
        return actions

    def update(self, states: torch.Tensor, critic: TD3Critic) -> None:
        loss = self.compute_loss(states, critic)
        self.gradient_descent_step(loss)

    def compute_loss(self, states: torch.Tensor, critic: TD3Critic) -> torch.Tensor:
        actions: torch.Tensor = self.forward(states)
        inputs = torch.cat([states, actions], dim=1)
        state_action_values: torch.Tensor = critic.forward(inputs)
        return -state_action_values.mean()

    def gradient_descent_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class TD3ActorMLP(MLPNetwork, TD3Actor):
    def __init__(
        self,
        shape: tuple,
        learning_rate: float = 3e-4,
        standard_deviation: float = 0.1,
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
        self.set_up_parameters(shape, learning_rate, standard_deviation)


class TD3ActorLSTM(LSTMNetwork, TD3Actor):
    def __init__(
        self,
        shape: tuple,
        learning_rate: float = 3e-4,
        standard_deviation: float = 0.1,
        output_activation=nn.Tanh,
        device_id: int = 0,
    ):
        super().__init__(
            shape, output_activation=output_activation, device_id=device_id
        )
        self.set_up_parameters(shape, learning_rate, standard_deviation)
