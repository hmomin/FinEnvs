import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .critic import Critic
from ..networks.generic_network import GenericNetwork
from ..networks.lstm import LSTMNetwork
from ..networks.multilayer_perceptron import MLPNetwork
from torch.distributions import Normal
from torch.optim import Adam
from typing import Tuple


class Actor(GenericNetwork):
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)

    def __new__(cls, *args, **kwargs):
        if cls is Actor:
            raise TypeError(
                f"'{cls.__name__}' should not be directly initiated. "
                + f"Try '{cls.__name__}MLP or {cls.__name__}LSTM' instead"
            )
        return super().__new__(cls)

    def set_up_parameters(
        self,
        shape: tuple,
        learning_rate: float,
        starting_alpha: float,
    ):
        self.learning_rate = learning_rate
        last_hidden_dim, num_actions = shape[-2:]
        self.mu_layer = nn.Linear(last_hidden_dim, num_actions, device=self.device)
        self.std_layer = nn.Linear(last_hidden_dim, num_actions, device=self.device)
        self.create_optimizer(learning_rate=learning_rate)
        self.log_alpha = torch.tensor(
            np.log(starting_alpha), device=self.device, requires_grad=True
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy: float = -self.num_actions

    def get_distribution(self, states: torch.Tensor):
        hiddens: torch.Tensor = self.forward(states)
        means: torch.Tensor = self.mu_layer(hiddens)
        std_devs: torch.Tensor = F.softplus(self.std_layer(hiddens))
        distribution = Normal(means, std_devs)
        return distribution

    def get_actions_and_log_probs(
        self, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self.get_distribution(states)
        normal_action = distribution.rsample()
        log_probs = distribution.log_prob(normal_action)
        reparameterized_actions = torch.tanh(normal_action)
        reparameterized_log_probs = log_probs - torch.log(
            1 - torch.tanh(normal_action).pow(2) + 1e-7
        )
        return reparameterized_actions, reparameterized_log_probs

    def compute_losses(
        self,
        states: torch.Tensor,
        critic_1: Critic,
        critic_2: Critic,
    ) -> torch.Tensor:
        actions, log_probs = self.get_actions_and_log_probs(states)
        mean_log_probs = log_probs.mean(dim=1, keepdim=True)
        entropy_term = -self.log_alpha.exp() * mean_log_probs
        states_actions = torch.cat([states, actions], dim=1)
        state_action_val_1 = critic_1.forward(states_actions)
        state_action_val_2 = critic_2.forward(states_actions)
        min_state_action_val = torch.min(state_action_val_1, state_action_val_2)
        loss = -(min_state_action_val + entropy_term).mean()
        alpha_loss = (
            -self.log_alpha.exp() * (mean_log_probs + self.target_entropy).detach()
        ).mean()
        return loss, alpha_loss

    def gradient_descent_step(
        self,
        policy_network_loss: torch.Tensor,
        alpha_loss: torch.Tensor,
    ) -> None:
        self.optimizer.zero_grad()
        policy_network_loss.backward()
        self.optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def update(
        self,
        states: torch.Tensor,
        critic_1: Critic,
        critic_2: Critic,
    ) -> None:
        loss, alpha_loss = self.compute_losses(states, critic_1, critic_2)
        self.gradient_descent_step(policy_network_loss=loss, alpha_loss=alpha_loss)


class ActorMLP(MLPNetwork, Actor):
    def __init__(
        self,
        shape: tuple,
        learning_rate=3e-4,
        starting_alpha=1.0,
        layer_activation=nn.ELU,
        output_activation=nn.Tanh,
        device_id: int = 0,
    ):
        network_shape = list(shape)
        self.num_actions = network_shape.pop()
        self.output_activation = output_activation
        super().__init__(network_shape, layer_activation, nn.Identity, device_id)
        self.set_up_parameters(shape, learning_rate, starting_alpha)


class ActorLSTM(LSTMNetwork, Actor):
    def __init__(
        self,
        shape: tuple,
        sequence_length: int,
        learning_rate=3e-4,
        starting_alpha=1.0,
        output_activation=nn.Tanh,
        device_id: int = 0,
    ):
        network_shape = list(shape)
        self.num_actions = network_shape.pop()
        self.output_activation = output_activation
        super().__init__(network_shape, sequence_length, nn.Identity, device_id)
        self.set_up_parameters(shape, learning_rate, starting_alpha)
