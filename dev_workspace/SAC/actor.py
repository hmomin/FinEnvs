import torch
import torch.nn as nn
import numpy as np
from finenvs.agents.networks.generic_network import GenericNetwork
from finenvs.agents.networks.lstm import LSTMNetwork
from finenvs.agents.networks.multilayer_perceptron import MLPNetwork
from critic import Critic
from torch.distributions import Normal
from torch.optim import Adam


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
    
    def set_up_parameters(self,
        shape:tuple,
        learning_rate:float,
        starting_std_dev:float,
        min_std_dev:float,
        max_std_dev: float,
        epsilon: float,
        alpha: float,
        alpha_learning_rate: float,
        target_entropy: float,
        ):
            self.shape = shape
            self.learning_rate = learning_rate
            self.min_std_dev = min_std_dev
            self.max_std_dev = max_std_dev
            self.create_optimizer(learning_rate=learning_rate)
            self.epsilon = epsilon
            self.log_alpha = torch.tensor(np.log(alpha))
            self.log_alpha.requires_grad = True
            self.alpha_optim = Adam([self.log_alpha],alpha_learning_rate)
            self.target_entropy = target_entropy
            self.log_std_dev = nn.Parameter(
                torch.ones((1,shape[-1]),device=self.device) * np.log(starting_std_dev)
            )
    
    def get_distribution(self,states:torch.Tensor):
        mean = self.forward(states)
        std_dev = torch.exp(self.log_std_dev)
        std_dev = torch.clamp(std_dev, self.min_std_dev, self.max_std_dev)
        distribution = Normal(mean,std_dev)
        return distribution
    
    def get_actions_and_log_probs(self,states: torch.Tensor):
        distribution = self.get_distribution(states)
        normal_action = distribution.rsample()
        log_probs = distribution.log_prob(normal_action)
        reparameterised_actions = torch.tanh(normal_action)
        reparameterised_log_probs  = log_probs - torch.log(1-\
            reparameterised_actions.pow(2)+self.epsilon)
        return reparameterised_actions,reparameterised_log_probs
    

    def compute_losses(
        self,
        states: torch.Tensor,
        critic_1: Critic,
        critic_2: Critic,
        ) -> torch.Tensor:
        actions, log_probs = self.get_actions_and_log_probs(states)
        entropy = self.log_alpha.exp() * log_probs
        state_action_val_1 = critic_1.forward(torch.cat([states,actions],dim=1))
        state_action_val_2 = critic_2.forward(torch.cat([states,actions],dim=1))
        min_state_action_val = torch.min(state_action_val_1, state_action_val_2)
        loss = -min_state_action_val - entropy
        alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
        return loss.mean(), alpha_loss
    
        
    def gradient_descent_step(self,
        policy_network_loss: torch.Tensor,
        alpha_loss: torch.Tensor,
        ) -> None:
        self.optimizer.zero_grad()
        policy_network_loss.backward()
        self.optimizer.step()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
    
    def update(self,
        states: torch.Tensor,
        critic_1: Critic,
        critic_2: Critic,
        ) -> None:
           loss, alpha_loss = self.compute_losses(states,critic_1,critic_2)
           self.gradient_descent_step(policy_network_loss=loss, alpha_loss=alpha_loss)

class ActorMLP(MLPNetwork,Actor):
    def __init__(
            self, 
            shape: tuple,
            learning_rate = 3e-4,
            min_std_dev = 1e-6,
            max_std_dev = 1,
            epsilon = 1e-7 ,
            alpha = 0.01,
            alpha_learning_rate = 0.001,
            target_entropy = -1.0,
            starting_std_dev = 1.0,
            layer_activation = nn.ELU,
            output_activation=nn.Tanh, 
            device_id: int = 0,
            ):
         super().__init__(shape, layer_activation, output_activation, device_id)
         self.set_up_parameters(shape,learning_rate,starting_std_dev,min_std_dev,max_std_dev,
            epsilon,alpha,alpha_learning_rate,target_entropy)
    
class ActorLSTM(LSTMNetwork,Actor):
    def __init__(
            self, 
            shape: tuple, 
            output_activation=nn.Tanh, 
            learning_rate = 3e-4,
            min_std_dev = 1e-6,
            max_std_dev = 1,
            epsilon = 1e-7 ,
            alpha = 0.01,
            alpha_learning_rate = 0.001,
            starting_std_dev = 1.0,
            target_entropy = -1.0,
            device_id: int = 0
            ):
         super().__init__(shape, output_activation, device_id)
         self.set_up_parameters(shape,learning_rate,starting_std_dev,min_std_dev,max_std_dev,
            epsilon,alpha,alpha_learning_rate,target_entropy)