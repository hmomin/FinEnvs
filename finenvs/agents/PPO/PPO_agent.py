import csv
import os
import torch
from ...base_object import BaseObject
from .buffer import Buffer
from .continuous_actor import ContinuousActor, ContinuousActorLSTM, ContinuousActorMLP
from .critic import Critic, CriticLSTM, CriticMLP
from datetime import datetime
from ...device_utils import set_device
from time import time
from typing import Tuple, Dict


class PPOAgent(BaseObject):
    def __init__(
        self,
        env_args: Dict,
        num_epochs: int,
        num_mini_batches: int,
        hidden_dims: Tuple[int],
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        write_to_csv: bool = True,
        device_id: int = 0,
    ):
        self.set_env_params(env_args)
        self.device = set_device(device_id)
        self.num_epochs = num_epochs
        self.clip_epsilon = clip_epsilon
        self.set_network_shapes(hidden_dims)
        self.buffer = Buffer(num_mini_batches, gamma, device_id)
        self.current_returns = torch.zeros(
            (self.num_envs,), device=self.device, requires_grad=False
        )
        self.training_returns = torch.zeros(
            (0, 1), device=self.device, requires_grad=False
        )
        self.evaluation_return = None
        self.num_samples = 0
        self.write_to_csv = write_to_csv
        if write_to_csv:
            self.create_progress_log()
        self.actor: ContinuousActor = None
        self.critic: Critic = None

    def __new__(cls, *args, **kwargs):
        if cls is PPOAgent:
            raise TypeError(
                "'PPOAgent' should not be directly instantiated. "
                + "Try 'PPOAgentMLP' or 'PPOAgentLSTM' instead."
            )
        return object.__new__(cls)

    def set_env_params(self, env_args: Dict) -> None:
        self.env_name: str = env_args["env_name"]
        self.num_envs: int = env_args["num_envs"]
        self.num_observations: int = env_args["num_observations"]
        self.num_actions: int = env_args["num_actions"]

    def set_network_shapes(self, hidden_dims: Tuple[int]) -> None:
        self.actor_shape = (self.num_observations, *hidden_dims, self.num_actions)
        self.critic_shape = (self.num_observations, *hidden_dims, 1)

    def create_progress_log(self) -> None:
        trials_dir = os.path.join(os.getcwd(), "trials")
        if not os.path.exists(trials_dir):
            os.mkdir(trials_dir)
        self.csv_name = os.path.join(
            trials_dir,
            datetime.now().strftime(f"{self.env_name}_PPO_%Y-%m-%d_%H-%M-%S.csv"),
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
        self, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = self.critic.forward(states).detach()
        distribution = self.actor.get_distribution(states)
        actions = torch.clamp(distribution.sample(), -1, +1)
        # the last environment is an "evaluation" environment, so it should strictly
        # use the means of the distribution
        evaluation_state = states[-1]
        evaluation_action = self.actor.forward(evaluation_state).detach()
        actions[-1, :] = evaluation_action
        log_probs = distribution.log_prob(actions).detach()
        return (actions, log_probs, values)

    def store(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self.buffer.store(states, actions, rewards, dones, log_probs, values)
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

    def log_progress(self) -> bool:
        self.num_samples += self.get_buffer_size()
        if self.evaluation_return == None:
            return False
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
        return True

    def train(self, current_states: torch.Tensor) -> int:
        current_state_values = self.critic.forward(current_states).detach()
        self.buffer.prepare_training_data(current_state_values)
        for _ in range(self.num_epochs):
            batch_dict = self.buffer.get_batches()
            batch_states = batch_dict["states"]
            batch_actions = batch_dict["actions"]
            batch_log_probs = batch_dict["log_probs"]
            batch_advantages = batch_dict["advantages"]
            batch_returns = batch_dict["returns"]
            mini_batch_list = self.buffer.get_mini_batch_indices()
            for mini_batch_indices in mini_batch_list:
                mini_batch_states = batch_states[mini_batch_indices, :]
                mini_batch_actions = batch_actions[mini_batch_indices, :]
                mini_batch_log_probs = batch_log_probs[mini_batch_indices, :]
                mini_batch_advantages = batch_advantages[mini_batch_indices, :]
                mini_batch_returns = batch_returns[mini_batch_indices, :]
                self.actor.gradient_ascent_step(
                    mini_batch_states,
                    mini_batch_actions,
                    mini_batch_log_probs,
                    mini_batch_advantages,
                )
                self.critic.gradient_descent_step(mini_batch_states, mini_batch_returns)
        progress_logged = self.log_progress()
        self.buffer.clear()
        return self.num_samples if progress_logged else 0


class PPOAgentMLP(PPOAgent):
    def __init__(
        self,
        env_args: Dict,
        num_epochs: int = 4,
        num_mini_batches: int = 4,
        learning_rate: float = 3e-4,
        starting_std_dev: float = 1.0,
        hidden_dims: Tuple[int] = (128, 128),
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        write_to_csv: bool = True,
        device_id: int = 0,
    ):
        super().__init__(
            env_args=env_args,
            num_epochs=num_epochs,
            num_mini_batches=num_mini_batches,
            hidden_dims=hidden_dims,
            clip_epsilon=clip_epsilon,
            gamma=gamma,
            write_to_csv=write_to_csv,
            device_id=device_id,
        )
        self.actor = ContinuousActorMLP(
            self.actor_shape,
            learning_rate,
            starting_std_dev,
            clip_epsilon,
            device_id=device_id,
        )
        self.critic = CriticMLP(self.critic_shape, learning_rate, device_id=device_id)


class PPOAgentLSTM(PPOAgent):
    def __init__(
        self,
        env_args: Dict,
        num_epochs: int = 4,
        num_mini_batches: int = 4,
        learning_rate: float = 3e-4,
        starting_std_dev: float = 1.0,
        hidden_dim: int = 128,
        clip_epsilon: float = 0.2,
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
            clip_epsilon=clip_epsilon,
            gamma=gamma,
            write_to_csv=write_to_csv,
            device_id=device_id,
        )
        self.actor = ContinuousActorLSTM(
            self.actor_shape,
            learning_rate,
            starting_std_dev,
            clip_epsilon,
            device_id=device_id,
        )
        self.critic = CriticLSTM(self.critic_shape, learning_rate, device_id=device_id)
