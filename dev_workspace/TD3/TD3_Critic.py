import imp
from math import gamma
from tkinter import Variable
import torch
import torch.nn as nn
from finevo.agents.networks.generic_network import GenericNetwork
from finevo.agents.networks.lstm import LSTMNetwork
from finevo.agents.networks.multilayer_perceptron import MLPNetwork

class TD3Critic(GenericNetwork):
    def __init__(self, device_id: int = 0):
        super().__init__(device_id=device_id)

    def __new__(cls, *args, **kwargs):
        if cls is TD3Critic:
            raise TypeError(
                f"'{cls.__name__}' should not be directly instantiated. "
                + f"Try '{cls.__name__}MLP' or '{cls.__name__}LSTM' instead."
            )
        return object.__new__(cls)

    def compute_critic_target(
        self, 
        rewards:torch.Tensor, 
        dones:torch.Tensor, 
        minimum_state_action_values_target:torch.Tensor,
        gamma:int,
        )->torch.Tensor:
        critic_target = rewards+(gamma)*(1-dones)*minimum_state_action_values_target
        return critic_target

    def compute_critic_loss(
        self, 
        rewards:torch.Tensor, 
        dones:torch.Tensor, 
        minimum_state_action_values_target:torch.Tensor,
        state_action_values: torch.Tensor,
        gamma:int,
    ) -> torch.Tensor:
        critic_target = \
            self.compute_critic_target(rewards, dones, minimum_state_action_values_target,gamma)
        squared_errors = (critic_target - state_action_values) ** 2
        mean_squared_error = squared_errors.mean()
        return mean_squared_error
    
    def gradient_descent_step(
        self, 
        rewards:torch.Tensor, 
        dones:torch.Tensor, 
        minimum_state_action_values_target:torch.Tensor,
        state_action_values: torch.Tensor,
        gamma:int):
        loss = self.compute_critic_loss(rewards, dones, minimum_state_action_values_target, 
                        state_action_values,gamma)
        loss.requires_grad_()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
class TD3CriticMLP(MLPNetwork, TD3Critic):
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


class TD3CriticLSTM(LSTMNetwork, TD3Critic):
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