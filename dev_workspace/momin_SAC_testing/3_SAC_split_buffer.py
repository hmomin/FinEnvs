import collections
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch.distributions import Normal
from typing import List, Tuple

# HYPERPARAMETERS
learning_rate = 3e-4  # for actor, critic, alpha
starting_alpha = 0.01
gamma = 0.99
minibatch_size = 100
max_size = 100_000
rho = 0.005  # for target network soft update
target_entropy = -1.0  # for automated alpha update


class ReplayBuffer:
    def __init__(self) -> None:
        self.states = collections.deque(maxlen=max_size)
        self.actions = collections.deque(maxlen=max_size)
        self.rewards = collections.deque(maxlen=max_size)
        self.next_states = collections.deque(maxlen=max_size)
        self.dones = collections.deque(maxlen=max_size)

    def put(
        self, transition: Tuple[np.ndarray, float, np.float64, np.ndarray, bool]
    ) -> None:
        (state, action, reward, next_state, done) = transition
        self.states.append(state)
        self.actions.append([action])
        self.rewards.append([reward])
        self.next_states.append(next_state)
        self.dones.append([0.0 if done else 1.0])

    def sample(
        self, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = random.sample(range(self.size()), n)
        states = [self.states[i] for i in indices]
        actions = [self.actions[i] for i in indices]
        rewards = [self.rewards[i] for i in indices]
        next_states = [self.next_states[i] for i in indices]
        dones = [self.dones[i] for i in indices]
        return (
            torch.tensor(states, dtype=torch.float),
            torch.tensor(actions, dtype=torch.float),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(next_states, dtype=torch.float),
            torch.tensor(dones, dtype=torch.float),
        )

    def size(self) -> int:
        return len(self.dones)


class Actor(nn.Module):
    def __init__(self, learning_rate: float) -> None:
        super(Actor, self).__init__()
        self.fully_connected_1 = nn.Linear(3, 128)
        self.fully_connected_mu = nn.Linear(128, 1)
        self.fully_connected_std = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.log_alpha = torch.tensor(np.log(starting_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = F.relu(self.fully_connected_1(inputs))
        mu: torch.Tensor = self.fully_connected_mu(inputs)
        std: torch.Tensor = F.softplus(self.fully_connected_std(inputs))
        distribution = Normal(mu, std)
        action: torch.Tensor = distribution.rsample()
        log_prob: torch.Tensor = distribution.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob: torch.Tensor = log_prob - torch.log(
            1 - torch.tanh(action).pow(2) + 1e-7
        )
        return real_action, real_log_prob

    def train_net(
        self,
        critic_1: "Critic",
        critic_2: "Critic",
        mini_batch: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> None:
        states, _, _, _, _ = mini_batch
        actions, log_probs = self.forward(states)
        entropy = -self.log_alpha.exp() * log_probs

        critic_1_val = critic_1.forward(states, actions)
        critic_2_val = critic_2.forward(states, actions)
        both_critic_values = torch.cat([critic_1_val, critic_2_val], dim=1)
        min_q = torch.min(both_critic_values, 1, keepdim=True)[0]

        loss = -min_q - entropy  # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(
            self.log_alpha.exp() * (log_probs + target_entropy).detach()
        ).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class Critic(nn.Module):
    def __init__(self, learning_rate: float) -> None:
        super(Critic, self).__init__()
        self.fully_connected_state = nn.Linear(3, 64)
        self.fully_connected_action = nn.Linear(1, 64)
        self.fully_connected_concat = nn.Linear(128, 32)
        self.fully_connected_out = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        hidden_1 = F.relu(self.fully_connected_state(state))
        hidden_2 = F.relu(self.fully_connected_action(action))
        both_hiddens = torch.cat([hidden_1, hidden_2], dim=1)
        q_value = F.relu(self.fully_connected_concat(both_hiddens))
        q_value: torch.Tensor = self.fully_connected_out(q_value)
        return q_value

    def train_net(
        self,
        target: torch.Tensor,
        mini_batch: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> None:
        states, actions, _, _, _ = mini_batch
        critic_values = self.forward(states, actions)
        loss = F.smooth_l1_loss(critic_values, target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, target_network: "Critic") -> None:
        for target_param, param in zip(target_network.parameters(), self.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - rho) + param.data * rho)


def calc_target(
    actor: "Actor",
    critic_1: "Critic",
    critic_2: "Critic",
    mini_batch: Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ],
) -> torch.Tensor:
    _, _, rewards, next_states, dones = mini_batch

    with torch.no_grad():
        next_actions, log_probs = actor.forward(next_states)
        entropy = -actor.log_alpha.exp() * log_probs
        critic_1_values = critic_1.forward(next_states, next_actions)
        critic_2_values = critic_2.forward(next_states, next_actions)
        both_critics = torch.cat([critic_1_values, critic_2_values], dim=1)
        min_q_values = torch.min(both_critics, 1, keepdim=True)[0]
        target = rewards + gamma * dones * (min_q_values + entropy)
    return target


def main():
    env = gym.make("Pendulum-v1")

    buffer = ReplayBuffer()
    actor = Actor(learning_rate)
    critic_1 = Critic(learning_rate)
    critic_2 = Critic(learning_rate)
    critic_1_target = deepcopy(critic_1)
    critic_2_target = deepcopy(critic_2)

    for episode_num in range(1000):
        score = 0.0
        state: np.ndarray = env.reset()
        done = False

        while not done:
            tensor_state = torch.from_numpy(state).float()
            action, _ = actor.forward(tensor_state)
            next_state, reward, done, info = env.step([2.0 * action.item()])
            buffer.put((state, action.item(), reward / 10.0, next_state, done))
            score += reward
            state = next_state

            if buffer.size() > minibatch_size:
                mini_batch = buffer.sample(minibatch_size)
                td_target = calc_target(
                    actor, critic_1_target, critic_2_target, mini_batch
                )
                critic_1.train_net(td_target, mini_batch)
                critic_2.train_net(td_target, mini_batch)
                actor.train_net(critic_1, critic_2, mini_batch)
                critic_1.soft_update(critic_1_target)
                critic_2.soft_update(critic_2_target)

        print(
            f"episode: {episode_num}\tscore: {score:.1f}\talpha: {actor.log_alpha.exp():.4f}"
        )

        if score > -100:
            break

    env.close()


if __name__ == "__main__":
    main()
