import csv
import os
import torch
from datetime import datetime
from finenvs.agents.SAC.actor import Actor, ActorLSTM, ActorMLP
from finenvs.agents.SAC.buffer import Buffer
from finenvs.agents.SAC.critic import Critic, CriticLSTM, CriticMLP
from finenvs.agents.networks.generic_network import GenericNetwork
from finenvs.base_object import BaseObject
from finenvs.device_utils import set_device
from time import time
from typing import Tuple, Dict


class SACAgent(BaseObject):
    def __init__(
        self,
        env_args: Dict,
        hidden_dims: Tuple[int],
        num_epochs: int = 1,
        mini_batch_size: int = 100,
        rho: float = 0.005,
        gamma: float = 0.99,
        write_to_csv: bool = True,
        device_id: int = 0,
    ) -> None:
        super().__init__()
        self.set_env_params(env_args)
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.rho = rho
        self.gamma = gamma
        self.write_to_csv = write_to_csv
        self.set_network_shapes(hidden_dims)
        self.device = set_device(device_id)
        self.buffer = Buffer(device_id=device_id)
        self.training_steps = 0
        self.current_returns = torch.zeros(
            (self.num_envs,), device=self.device, requires_grad=False
        )
        self.training_returns = torch.zeros(
            (0, 1), device=self.device, requires_grad=False
        )
        self.evaluation_return = None
        self.num_samples = 0
        if self.write_to_csv:
            self.create_progress_log()
        self.actor: Actor = None
        self.critic_1: Critic = None
        self.critic_2: Critic = None
        self.target_critic_1: Critic = None
        self.target_critic_2: Critic = None

    def __new__(cls, *args, **kwargs):
        if cls is SACAgent:
            raise TypeError(
                "'SACAgent' should not be directly instantiated. "
                + "Try 'SACAgentMLP' or 'SACAgentLSTM' instead."
            )
        return object.__new__(cls)

    def set_env_params(self, env_args: Dict) -> None:
        self.env_name: str = env_args["env_name"]
        self.num_envs: str = env_args["num_envs"]
        self.num_observations: int = env_args["num_observations"]
        self.num_actions: int = env_args["num_actions"]

    def set_network_shapes(self, hidden_dims: Tuple[int]) -> None:
        self.actor_shape = (self.num_observations, *hidden_dims, self.num_actions)
        self.critic_shape = (self.num_observations + self.num_actions, *hidden_dims, 1)

    def initialize_target_parameters(self) -> None:
        original_networks: "list[GenericNetwork]" = [self.critic_1, self.critic_2]
        target_networks: "list[GenericNetwork]" = [
            self.target_critic_1,
            self.target_critic_2,
        ]
        with torch.no_grad():
            for net, target_net in zip(original_networks, target_networks):
                for parameter, target_parameter in zip(
                    net.parameters(), target_net.parameters()
                ):
                    target_parameter.sub_(target_parameter)
                    target_parameter.add_(parameter)

    def create_progress_log(self) -> None:
        trials_dir = os.path.join(os.getcwd(), "trials")
        if not os.path.exists(trials_dir):
            os.mkdir(trials_dir)
        self.csv_name = os.path.join(
            trials_dir,
            datetime.now().strftime(f"{self.env_name}_SAC_%Y-%m-%d-%H-%M-%S.csv"),
        )
        csv_fields = [
            "unix_time",
            "num_training_samples",
            "evaluation_return",
            "num_training_episodes",
            "mean_training_return",
            "std_dev_training_return",
        ]
        with open(self.csv_name, "a") as f:
            writer = csv.writer(f)
            writer.writerow(csv_fields)

    def get_buffer_size(self) -> int:
        return self.buffer.size()

    def step(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            dist = self.actor.get_distribution(states)
            normal_action = dist.rsample()
            actions = torch.tanh(normal_action).detach()
            # The last environiment is an "evaluation" environiment, so it should strictly
            # use the means of the distribution
            actions[-1, :] = dist.loc[-1, :]
        return actions

    def store(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.buffer.store(states, actions, rewards, next_states, dones)
        self.num_samples += self.num_envs
        self.current_returns += rewards
        evaluation_done = dones[-1]
        training_dones = dones[:-1]
        training_done_indices = (training_dones == 1).nonzero()
        finished_training_returns = self.current_returns[training_done_indices]
        self.training_returns = torch.cat(
            [self.training_returns, finished_training_returns], dim=0
        )
        if evaluation_done.item() == 1:
            self.evaluation_return = self.current_returns[-1].item()
            self.current_returns[-1] = 0
        self.current_returns[training_done_indices] = 0

    def log_progress(self) -> float:
        if self.evaluation_return == None:
            return None
        evaluation_return = self.evaluation_return
        num_training_episodes = self.training_returns.shape[0]
        mean_training_return = self.training_returns.mean().item()
        std_dev_training_return = self.training_returns.std().item()
        record_fields = [
            time(),
            self.num_samples,
            evaluation_return,
            num_training_episodes,
            mean_training_return,
            std_dev_training_return,
        ]
        if self.write_to_csv:
            with open(self.csv_name, "a") as f:
                writer = csv.writer(f)
                writer.writerow(record_fields)
        print(
            f"num samples: {self.num_samples} - "
            + f"evaluation return: {evaluation_return:.6f} - "
            + f"mean training return: {mean_training_return:.6f} - "
            + f"std dev training return: {std_dev_training_return:.6f}"
        )
        self.training_returns = torch.zeros(
            (0, 1), device=self.device, requires_grad=False
        )
        self.evaluation_return = None
        return evaluation_return

    def train(self) -> Tuple[int, float]:
        for _ in range(self.num_epochs):
            mini_batch_dict: Dict[str, torch.Tensor] = self.buffer.get_mini_batch(
                self.mini_batch_size
            )
            states = mini_batch_dict["states"]
            actions = mini_batch_dict["actions"]
            rewards = mini_batch_dict["rewards"]
            next_states = mini_batch_dict["next_states"]
            dones = mini_batch_dict["dones"]
            targets = self.compute_targets(rewards, next_states, dones)
            self.critic_1.update(states, actions, targets)
            self.critic_2.update(states, actions, targets)
            self.actor.update(states, self.critic_1, self.critic_2)
            self.update_target_networks()
        evaluation_return = self.log_progress()
        self.training_steps += 1
        return (
            (self.num_samples, evaluation_return)
            if evaluation_return != None
            else (0, 0)
        )

    def compute_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            actions, log_probs = self.actor.get_actions_and_log_probs(next_states)
            next_state_action_value_1 = self.target_critic_1.forward(
                torch.cat([next_states, actions], dim=1)
            )
            next_state_action_value_2 = self.target_critic_2.forward(
                torch.cat([next_states, actions], dim=1)
            )
            min_state_action_value = torch.min(
                next_state_action_value_1, next_state_action_value_2
            )
            entropy = -self.actor.log_alpha.exp() * log_probs
            target = rewards + self.gamma * (1 - dones) * (
                min_state_action_value - entropy
            )
        return target

    def update_target_networks(self) -> None:
        self.update_target_network_params(self.critic_1, self.target_critic_1)
        self.update_target_network_params(self.critic_2, self.target_critic_2)

    def update_target_network_params(
        self,
        net: GenericNetwork,
        target_net: GenericNetwork,
    ) -> None:
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.rho) + param.data * self.rho
            )


class SACAgentMLP(SACAgent):
    def __init__(
        self,
        env_args: Dict,
        hidden_dims: Tuple[int] = (3, 3),
        learning_rate: float = 3e-4,
        num_epochs: int = 1,
        mini_batch_size: int = 100,
        starting_alpha: float = 1.0,
        rho: float = 0.005,
        gamma: float = 0.99,
        write_to_csv: bool = True,
        device_id: int = 0,
    ) -> None:
        super().__init__(
            env_args,
            hidden_dims,
            num_epochs,
            mini_batch_size,
            rho,
            gamma,
            write_to_csv,
            device_id,
        )

        self.actor = ActorMLP(
            self.actor_shape, learning_rate, starting_alpha, device_id=device_id
        )
        self.critic_1 = CriticMLP(self.critic_shape, learning_rate, device_id=device_id)
        self.critic_2 = CriticMLP(self.critic_shape, learning_rate, device_id=device_id)
        self.target_critic_1 = CriticMLP(
            self.critic_shape, learning_rate, device_id=device_id
        )
        self.target_critic_2 = CriticMLP(
            self.critic_shape, learning_rate, device_id=device_id
        )
        self.initialize_target_parameters()


class SACAgentLSTM(SACAgent):
    def __init__(
        self,
        env_args: Dict,
        hidden_dim: int = 128,
        learning_rate=3e-4,
        num_epochs: int = 1,
        mini_batch_size: int = 100,
        starting_alpha: float = 1.0,
        rho: float = 0.005,
        gamma: float = 0.99,
        write_to_csv: bool = True,
        device_id: int = 0,
    ) -> None:
        hidden_dims = (hidden_dim,)
        super().__init__(
            env_args,
            hidden_dims,
            num_epochs,
            mini_batch_size,
            rho,
            gamma,
            write_to_csv,
            device_id,
        )
        self.actor = ActorLSTM(
            self.actor_shape, learning_rate, starting_alpha, device_id=device_id
        )
        self.critic_1 = CriticLSTM(
            self.critic_shape, learning_rate, device_id=device_id
        )
        self.critic_2 = CriticLSTM(
            self.critic_shape, learning_rate, device_id=device_id
        )
        self.target_critic_1 = CriticLSTM(
            self.critic_shape, learning_rate, device_id=device_id
        )
        self.target_critic_2 = CriticLSTM(
            self.critic_shape, learning_rate, device_id=device_id
        )
        self.initialize_target_parameters()
