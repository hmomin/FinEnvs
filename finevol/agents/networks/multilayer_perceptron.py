import numpy as np
import torch
import torch.nn as nn
from ...device_utils import set_device
from torch.distributions.normal import Normal
from torch.optim import Adam

torch.autograd.set_detect_anomaly(True)


class MLPNetwork(nn.Module):
    def __init__(
        self,
        shape: tuple,
        layer_activation=nn.ELU,
        output_activation=nn.Identity,
        device_id: int = 0,
    ):
        super(MLPNetwork, self).__init__()
        self.device = set_device(device_id)
        self.create_layers(shape, layer_activation, output_activation)

    def create_layers(self, shape: tuple, layer_activation, output_activation) -> None:
        layers = []
        for idx in range(len(shape) - 1):
            activation = layer_activation if idx < len(shape) - 2 else output_activation
            layers.append(nn.Linear(shape[idx], shape[idx + 1], device=self.device))
            layers.append(activation())
        self.network = nn.Sequential(*layers)

    def create_optimizer(self, learning_rate):
        self.optimizer = Adam(self.parameters(), learning_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ContinuousActor(MLPNetwork):
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
            shape=shape,
            layer_activation=layer_activation,
            output_activation=output_activation,
            device_id=device_id,
        )
        self.clip_epsilon = clip_epsilon
        num_actions = shape[-1]
        self.log_standard_deviation = nn.Parameter(
            torch.ones((1, num_actions), device=self.device) * np.log(starting_std_dev)
        )
        self.create_optimizer(learning_rate)

    def get_distribution(self, inputs: torch.Tensor) -> Normal:
        means = self.forward(inputs)
        standard_deviations = torch.exp(self.log_standard_deviation)
        return Normal(means, standard_deviations)

    def get_new_log_probs(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        distribution = self.get_distribution(states)
        new_log_probs = distribution.log_prob(actions)
        return new_log_probs

    def gradient_ascent_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> None:
        self.optimizer.zero_grad()
        loss = self.compute_actor_loss(states, actions, log_probs, advantages)
        loss.backward()
        self.optimizer.step()

    def compute_actor_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        new_log_probs = self.get_new_log_probs(states, actions)
        prob_ratios = torch.exp(new_log_probs - log_probs)
        first_term = prob_ratios * advantages
        second_term = (
            torch.clamp(prob_ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages
        )
        clip_objective = torch.minimum(first_term, second_term)
        mean_clipped_objective = clip_objective.mean().mean()
        return -mean_clipped_objective


class Critic(MLPNetwork):
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
