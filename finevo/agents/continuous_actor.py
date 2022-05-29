import numpy as np
import torch
import torch.nn as nn
from .networks.generic_network import GenericNetwork
from .networks.lstm import LSTMNetwork
from .networks.multilayer_perceptron import MLPNetwork
from torch.distributions.normal import Normal


class ContinuousActor(GenericNetwork):
    def __init__(self, device_id: int = 0):
        super().__init__(device_id=device_id)

    def __new__(cls, *args, **kwargs):
        if cls is ContinuousActor:
            raise TypeError(
                f"'{cls.__name__}' should not be directly instantiated. "
                + f"Try '{cls.__name__}MLP' or '{cls.__name__}LSTM' instead."
            )
        return object.__new__(cls)

    def set_up_parameters(
        self,
        shape: tuple,
        learning_rate: float,
        starting_std_dev: float,
        clip_epsilon: float,
    ):
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


class ContinuousActorMLP(MLPNetwork, ContinuousActor):
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
        self.set_up_parameters(shape, learning_rate, starting_std_dev, clip_epsilon)


class ContinuousActorLSTM(LSTMNetwork, ContinuousActor):
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
            shape=shape, output_activation=output_activation, device_id=device_id,
        )
        self.set_up_parameters(shape, learning_rate, starting_std_dev, clip_epsilon)
