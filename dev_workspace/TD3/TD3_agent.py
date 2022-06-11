import csv
from math import gamma
import os
import torch
from finevo.agents.networks.generic_network import GenericNetwork
from finevo.base_object import BaseObject
from TD3_buffer import Buffer
from TD3_Critic import TD3Critic, TD3CriticLSTM, TD3CriticMLP
from TD3_actor import TD3Actor, TD3ActorLSTM, TD3ActorMLP
from datetime import datetime
from finevo.device_utils import set_device
from time import time
from typing import Tuple, Dict

class TD3Agent(BaseObject):
    def __init__(
        self,
        env_args:Dict,
        num_epochs:int,
        num_mini_batches:int,
        hidden_dims:Tuple[int],
        policy_delay:int = 2,
        gamma:float=0.99,
        rho:float = 0.055,
        write_to_csv:bool=True,
        device_id:int=0,
        ) -> None:
        self.set_env_params(env_args)
        self.device = set_device(device_id)
        self.num_epochs = num_epochs
        self.policy_delay = policy_delay
        self.rho = rho
        self.gamma = gamma
        self.set_network_shapes(hidden_dims)
        self.buffer = Buffer(num_mini_batches,self.gamma,device_id)
        self.current_returns = torch.zeros(
            (self.num_envs,),device=self.device,requires_grad=False
        )
        self.training_returns = torch.zeros(
            (0, 1), device=self.device, requires_grad=False
        )
        self.evaluation_return = None
        self.num_samples = 0
        self.write_to_csv = write_to_csv
        if self.write_to_csv:
            self.create_progress_log()
        self.actor:TD3Actor = None
        self.actor_target:TD3Actor = None
        self.critic_1:TD3Critic = None
        self.critic_2:TD3Critic = None
        self.critic_target_1:TD3Critic = None
        self.critic_target_2: TD3Critic = None

    def __new__(cls, *args, **kwargs):
        if cls is TD3Agent:
            raise TypeError(
                "'TD3Agent' should not be directly instantiated. "
                + "Try 'TD3AgentMLP' or 'TD3AgentLSTM' instead."
            )
        return object.__new__(cls)

    def set_env_params(self,env_args:Dict)->None:
        self.env_name:str = env_args["env_name"]
        self.num_envs:int = env_args["num_envs"]
        self.num_observations:int = env_args["num_observations"]
        self.num_actions:int = env_args["num_actions"]
    
    def set_network_shapes(self,hidden_dims:Tuple[int])->None:
        self.actor_shape = (self.num_observations,*hidden_dims,1)
        self.critic_shape = (self.num_observations+1,*hidden_dims,1)
    
    def create_progress_log(self)->None:
        trails_dir = os.path.join(os.getcwd(),"trails")
        if not os.path.exists(trails_dir):
            os.mkdir(trails_dir)
        self.csv_name = os.path.join(
            trails_dir,
            datetime.now().strftime(f"{self.env_name}_TD3_%Y-%m-%d-%H-%M-%S.csv"),
        )
        csv_fields = [
            "unix_time",
            "num_training_samples",
            "evaluation_return",
            "num_training_episodes",
            "mean_training_return",
            "std_dev_training_return"
        ]
        with open(self.csv_name,"a") as f:
            writer = csv.writer(f)
            writer.writerow(csv_fields)
    
    def get_buffer_size(self)->int:
        return self.buffer.size()
    
    def step(
        self, 
        states:torch.Tensor,
    )->Tuple[torch.Tensor,torch.Tensor]:
        actions = self.actor.take_noised_action(states)
        state_action_values_1 = self.critic_1.forward(torch.cat([states,actions],dim=1)).detach()
        state_action_values_2 = self.critic_2.forward(torch.cat([states,actions],dim=1)).detach()

        # the last environment is an "evaluation" environment, so it should strictly
        # use the means of the distribution
        evaluation_state = states[-1]
        evaluation_action = self.actor.forward(evaluation_state).detach()
        actions[-1,:] = evaluation_action
        return (actions,  state_action_values_1, state_action_values_2)
    

    
    def store(
        self,
        states:torch.Tensor,
        actions:torch.Tensor,
        rewards:torch.Tensor,
        dones:torch.Tensor,
        next_states:torch.Tensor,
    )->None:
        self.buffer.store(states,actions,rewards,dones,next_states)
        self.current_returns+=rewards
        evaluation_done = dones[-1]
        training_dones = dones[:-1]
        training_done_indices = (training_dones==1).nonzero()
        finished_training_returns = self.current_returns[training_done_indices]
        self.training_returns = torch.cat(
            [self.training_returns,finished_training_returns],dim=0
        )
        if evaluation_done.item()==1:
            self.evaluation_return = self.current_returns[-1].item()
            self.current_returns[-1]=0
        self.current_returns[training_done_indices]=0

    def log_progress(self)->bool:
        self.num_samples += self.buffer.size()
        if self.evaluation_return == None:
            return False
        evaluation_return = self.evaluation_return
        num_training_episodes =  self.training_returns.shape[0]
        mean_training_return = self.training_returns.mean().item()
        std_dev_training_return = self.training_returns.std().item()
        record_fields = [
            time(),
            self.num_samples,
            evaluation_return,
            num_training_episodes,
            mean_training_return,
            std_dev_training_return
        ]
        if self.write_to_csv:
            with open(self.csv_name,"a") as f:
                writer = csv.writer(f)
                writer.writerow(record_fields)
        print(
            f"num samples: {self.num_samples} - "
            + f"evaluation return: {evaluation_return:.6f} - "
            + f"mean training return: {mean_training_return:.6f} - "
            + f"std dev training return: {std_dev_training_return:.6f}"
        )
        self.training_returns = torch.zeros(
            (0,1), device = self.device, requires_grad=False
        )
        self.evaluation_return = None
        return True
    
    def update_target_network_params(self, 
    net:GenericNetwork, 
    target_net: GenericNetwork
    ) -> None:
        for target_param, param in zip(target_net.parameters(),net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.rho) + param.data * self.rho)


    def train(self, current_states:torch.Tensor)->int:
        self.buffer.prepare_training_data()
        for _ in range(self.num_epochs):
            batch_dict = self.buffer.get_batches()
            batch_states = batch_dict["states"]
            batch_actions = batch_dict["actions"]
            batch_rewards = batch_dict["rewards"]
            batch_dones = batch_dict["dones"]
            batch_next_states = batch_dict["next_states"]
            mini_batch_list = self.buffer.get_mini_batch_indices()
            for iteration_num,mini_batch_indices in enumerate(mini_batch_list):
                mini_batch_states = batch_states[mini_batch_indices, :]
                mini_batch_actions = batch_actions[mini_batch_indices, :]
                mini_batch_rewards = batch_rewards[mini_batch_indices, :]
                mini_batch_dones = batch_dones[mini_batch_indices, :]
                mini_batch_next_states = batch_next_states[mini_batch_indices, :]
                mini_batch_target_actions = self.actor_target.take_noised_action(mini_batch_next_states)
                mini_batch_state_action_values_1 = self.critic_1.forward(torch.cat([mini_batch_states,mini_batch_actions],dim=1)).detach()
                mini_batch_state_action_values_2 = self.critic_2.forward(torch.cat([mini_batch_states,mini_batch_actions],dim=1)).detach()
                mini_batch_state_action_values_target_1 = self.critic_target_1.forward(torch.cat([mini_batch_next_states,mini_batch_target_actions],dim=1)).detach()
                mini_batch_state_action_values_target_2 = self.critic_target_2.forward(torch.cat([mini_batch_next_states,mini_batch_target_actions],dim=1)).detach()

                mini_batch_min_state_action_values_target = torch.minimum(
                    mini_batch_state_action_values_target_1,mini_batch_state_action_values_target_2
                )

                self.critic_1.gradient_descent_step(mini_batch_rewards, mini_batch_dones, 
                    mini_batch_min_state_action_values_target, 
                    mini_batch_state_action_values_1,self.gamma)
                self.critic_2.gradient_descent_step(mini_batch_rewards, mini_batch_dones, 
                    mini_batch_min_state_action_values_target, 
                    mini_batch_state_action_values_2,self.gamma)
                if iteration_num%self.policy_delay == 0:
                    mini_batch_actions_without_noise = self.actor.forward(mini_batch_states)
                    mini_batch_state_action_values_actor = self.critic_1.forward(torch.cat([
                        mini_batch_states, mini_batch_actions_without_noise
                    ],dim=1))

                    self.actor.gradient_ascent_step(mini_batch_state_action_values_actor)
                    self.update_target_network_params(self.critic_1, self.critic_target_1)
                    self.update_target_network_params(self.critic_2, self.critic_target_2)
                    self.update_target_network_params(self.actor, self.actor_target)




        progress_logged = self.log_progress()
        self.buffer.clear()
        return self.num_samples if progress_logged else 0
    
class TD3AgentMLP(TD3Agent):
    def __init__(
        self,
        env_args:Dict,
        num_epochs:int =4,
        num_mini_batches:int =4,
        learning_rate:float = 3e-5,
        standard_deviation:float = 0.1,
        hidden_dims:Tuple[int] = (128,128),
        policy_delay:int = 2,
        gamma:float=0.99,
        rho:float = 0.055,
        write_to_csv:bool=True,
        device_id:int=0,
        ) -> None:
        super().__init__(
            env_args=env_args, 
            num_epochs=num_epochs, 
            num_mini_batches=num_mini_batches, 
            hidden_dims=hidden_dims, 
            gamma=gamma, 
            policy_delay=policy_delay,
            rho=rho,
            write_to_csv=write_to_csv, 
            device_id=device_id,
            )
        self.actor = TD3ActorMLP(
            self.actor_shape,
            learning_rate,
            standard_deviation,
            device_id=device_id
        )
        self.actor_target = TD3ActorMLP(
            self.actor_shape,
            learning_rate,
            standard_deviation,
            device_id=device_id
        )
        self.critic_1 = TD3CriticMLP(self.critic_shape,learning_rate,device_id=device_id)
        self.critic_2 = TD3CriticMLP(self.critic_shape,learning_rate,device_id=device_id)
        self.critic_target_1 = TD3CriticMLP(self.critic_shape,learning_rate,device_id=device_id)
        self.critic_target_2 = TD3CriticMLP(self.critic_shape,learning_rate,device_id=device_id)

class TD3AgentLSTM(TD3Agent):
    def __init__(
        self,
        env_args: Dict,
        num_epochs: int = 4,
        num_mini_batches: int = 4,
        learning_rate: float = 3e-5,
        standard_deviation:float = 0.1,
        hidden_dim: int = 128,
        policy_delay:int = 2,
        rho:float = 0.055,
        gamma: float = 0.99,
        write_to_csv: bool = True,
        device_id: int = 0,
    ):
        hidden_dims = (hidden_dim,)
        super().__init__(
            env_args=env_args,
            num_epochs=num_epochs,
            num_mini_batches=num_mini_batches,
            hidden_dims=hidden_dims,
            gamma=gamma,
            policy_delay=policy_delay,
            rho = rho,
            write_to_csv=write_to_csv,
            device_id=device_id,
        )
        self.actor = TD3ActorLSTM(
            self.actor_shape,
            learning_rate,
            standard_deviation,
            device_id=device_id,
        )
        self.actor_target = TD3ActorLSTM(
            self.actor_shape,
            learning_rate,
            standard_deviation,
            device_id=device_id,
        )
        self.critic_1 = TD3CriticLSTM(self.critic_shape, learning_rate, device_id=device_id)
        self.critic_2 = TD3CriticLSTM(self.critic_shape, learning_rate, device_id=device_id)
        self.critic_target_1 = TD3CriticLSTM(self.critic_shape, learning_rate, device_id=device_id)
        self.critic_target_2 = TD3CriticLSTM(self.critic_shape, learning_rate, device_id=device_id)