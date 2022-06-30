import csv
import os
import torch
from datetime import datetime
from .actor import Actor, ActorLSTM, ActorMLP
from .buffer import Buffer
from .critic import Critic, CriticLSTM, CriticMLP
from ..networks.generic_network import GenericNetwork
from ...base_object import BaseObject
from ...device_utils import set_device
from time import time
from typing import Tuple, Dict


class TD3Agent(BaseObject):
    def __init__(
        self,
        env_args: Dict,
        hidden_dims: Tuple[int],
        num_epochs: int = 1,
        mini_batch_size: int = 100,
        training_std_dev: int = 0.2,
        training_clip: int = 0.5,
        rho: float = 0.005,
        gamma: float = 0.99,
        policy_delay: int = 2,
        model_save_interval: int = 20,
        write_to_csv: bool = True,
        device_id: int = 0,
    ) -> None:
        self.set_env_params(env_args)
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.training_std_dev = training_std_dev
        self.traning_clip = training_clip
        self.rho = rho
        self.gamma = gamma
        self.policy_delay = policy_delay
        self.write_to_csv = write_to_csv
        self.set_network_shapes(hidden_dims)
        self.device = set_device(device_id)
        max_size = 250_000 if self.num_observations > 1000 else 1_000_000
        self.buffer = Buffer(max_size=max_size, device_id=device_id)
        self.training_steps = 0
        self.current_returns = torch.zeros(
            (self.num_envs,), device=self.device, requires_grad=False
        )
        self.training_returns = torch.zeros(
            (0, 1), device=self.device, requires_grad=False
        )
        self.evaluation_return = None
        self.num_samples = 0
        self.num_steps = 0
        self.save_interval = model_save_interval
        self.designate_trials_dir()
        if self.write_to_csv:
            self.create_progress_log()
        self.actor: Actor = None
        self.target_actor: Actor = None
        self.critic_1: Critic = None
        self.critic_2: Critic = None
        self.target_critic_1: Critic = None
        self.target_critic_2: Critic = None

    def __new__(cls, *args, **kwargs):
        if cls is TD3Agent:
            raise TypeError(
                "'TD3Agent' should not be directly instantiated. "
                + "Try 'TD3AgentMLP' or 'TD3AgentLSTM' instead."
            )
        return object.__new__(cls)

    def set_env_params(self, env_args: Dict) -> None:
        self.env_name: str = env_args["env_name"]
        self.num_envs: int = env_args["num_envs"]
        self.num_observations: int = env_args["num_observations"]
        self.num_actions: int = env_args["num_actions"]

    def set_network_shapes(self, hidden_dims: Tuple[int]) -> None:
        self.actor_shape = (self.num_observations, *hidden_dims, self.num_actions)
        self.critic_shape = (self.num_observations + self.num_actions, *hidden_dims, 1)

    def initialize_target_parameters(self) -> None:
        original_networks: "list[GenericNetwork]" = [
            self.actor,
            self.critic_1,
            self.critic_2,
        ]
        target_networks: "list[GenericNetwork]" = [
            self.target_actor,
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

    def designate_trials_dir(self) -> None:
        timestamped_name = datetime.now().strftime(
            f"{self.env_name}_TD3_%Y-%m-%d_%H-%M-%S"
        )
        trials_dir = os.path.join(os.getcwd(), "trials", timestamped_name)
        self.trials_dir = trials_dir
        self.csv_name = os.path.join(
            trials_dir,
            f"{timestamped_name}.csv",
        )

    def create_trials_dir(self) -> None:
        if not os.path.exists(self.trials_dir):
            os.mkdir(self.trials_dir)

    def create_progress_log(self) -> None:
        self.create_trials_dir()
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
        actions = self.actor.get_noisy_actions(states)
        # the last environment is an "evaluation" environment, so it should strictly
        # use the means of the distribution
        evaluation_state = states[-1]
        evaluation_action = self.actor.forward(evaluation_state.float()).detach()
        actions[-1, :] = evaluation_action
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
        self.num_steps += 1
        if self.save_interval > 0 and self.num_steps % self.save_interval == 0:
            self.create_trials_dir()
            self.save_model()
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
            for key in mini_batch_dict.keys():
                mini_batch_dict[key].requires_grad_()
            states = mini_batch_dict["states"]
            actions = mini_batch_dict["actions"]
            rewards = mini_batch_dict["rewards"]
            next_states = mini_batch_dict["next_states"]
            dones = mini_batch_dict["dones"]
            targets = self.compute_targets(rewards, next_states, dones)
            self.critic_1.update(states, actions, targets, True)
            self.critic_2.update(states, actions, targets, False)
            if self.training_steps % self.policy_delay == 0:
                self.actor.update(states, self.critic_1)
                self.update_target_networks()
        evaluation_return = self.log_progress()
        self.training_steps += 1
        return (
            (self.num_samples, evaluation_return)
            if evaluation_return != None
            else (0, 0)
        )

    def compute_targets(
        self, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        float_next_states = next_states.float()
        target_actions: torch.Tensor = self.target_actor.forward(float_next_states)
        noise = (
            torch.randn(target_actions.shape, device=self.device)
            * self.training_std_dev
        )
        clipped_noise = torch.clamp(noise, -self.traning_clip, +self.traning_clip)
        target_actions = torch.clamp(target_actions + clipped_noise, -1, +1)
        inputs = torch.cat([float_next_states, target_actions], dim=1)
        target_state_action_values_1 = self.target_critic_1.forward(inputs)
        target_state_action_values_2 = self.target_critic_2.forward(inputs)
        target_state_action_values = torch.minimum(
            target_state_action_values_1, target_state_action_values_2
        )
        return rewards + self.gamma * (1 - dones) * target_state_action_values

    def update_target_networks(self) -> None:
        self.update_target_network_params(self.actor, self.target_actor)
        self.update_target_network_params(self.critic_1, self.target_critic_1)
        self.update_target_network_params(self.critic_2, self.target_critic_2)

    def update_target_network_params(
        self, net: GenericNetwork, target_net: GenericNetwork
    ) -> None:
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.rho) + param.data * self.rho
            )

    def save_model(self) -> None:
        saved_model_path = os.path.join(
            self.trials_dir, f"{self.env_name}_TD3_{self.num_steps}.pth"
        )
        torch.save(self.actor.state_dict(), saved_model_path)


class TD3AgentMLP(TD3Agent):
    def __init__(
        self,
        env_args: Dict,
        hidden_dims: Tuple[int] = (128, 128),
        learning_rate: float = 3e-4,
        num_epochs: int = 1,
        mini_batch_size: int = 100,
        action_std_dev: float = 0.1,
        training_std_dev: float = 0.2,
        training_clip: float = 0.5,
        rho: float = 0.005,
        gamma: float = 0.99,
        policy_delay: int = 2,
        model_save_interval: int = 20,
        write_to_csv: bool = True,
        device_id: int = 0,
    ) -> None:
        super().__init__(
            env_args=env_args,
            hidden_dims=hidden_dims,
            num_epochs=num_epochs,
            mini_batch_size=mini_batch_size,
            training_std_dev=training_std_dev,
            training_clip=training_clip,
            rho=rho,
            gamma=gamma,
            policy_delay=policy_delay,
            model_save_interval=model_save_interval,
            write_to_csv=write_to_csv,
            device_id=device_id,
        )
        self.actor = ActorMLP(
            self.actor_shape, learning_rate, action_std_dev, device_id=device_id
        )
        self.critic_1 = CriticMLP(self.critic_shape, learning_rate, device_id=device_id)
        self.critic_2 = CriticMLP(self.critic_shape, learning_rate, device_id=device_id)
        self.target_actor = ActorMLP(
            self.actor_shape, learning_rate, action_std_dev, device_id=device_id
        )
        self.target_critic_1 = CriticMLP(
            self.critic_shape, learning_rate, device_id=device_id
        )
        self.target_critic_2 = CriticMLP(
            self.critic_shape, learning_rate, device_id=device_id
        )
        self.initialize_target_parameters()


class TD3AgentLSTM(TD3Agent):
    def __init__(
        self,
        env_args: Dict,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        num_epochs: int = 1,
        mini_batch_size: int = 100,
        action_std_dev: float = 0.1,
        training_std_dev: float = 0.2,
        training_clip: float = 0.5,
        rho: float = 0.005,
        gamma: float = 0.99,
        policy_delay: int = 2,
        model_save_interval: int = 20,
        write_to_csv: bool = True,
        device_id: int = 0,
    ):
        hidden_dims = (hidden_dim,)
        super().__init__(
            env_args=env_args,
            hidden_dims=hidden_dims,
            num_epochs=num_epochs,
            mini_batch_size=mini_batch_size,
            training_std_dev=training_std_dev,
            training_clip=training_clip,
            rho=rho,
            gamma=gamma,
            policy_delay=policy_delay,
            model_save_interval=model_save_interval,
            write_to_csv=write_to_csv,
            device_id=device_id,
        )
        self.actor = ActorLSTM(
            self.actor_shape,
            learning_rate,
            action_std_dev,
            device_id=device_id,
        )
        self.target_actor = ActorLSTM(
            self.actor_shape,
            learning_rate,
            action_std_dev,
            device_id=device_id,
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
