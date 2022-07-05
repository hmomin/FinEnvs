import torch
import unittest
from finenvs.agents.TD3.actor import ActorMLP, ActorLSTM
from finenvs.agents.TD3.critic import CriticMLP, CriticLSTM
from finenvs.device_utils import set_device


class TestActorCritic(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.num_envs = 32
        self.num_observations = 24
        self.num_actions = 3
        self.sequence_length = 5
        self.device = set_device(0)
        self.mlp_actor = ActorMLP((self.num_observations, 128, 128, self.num_actions))
        self.lstm_actor = ActorLSTM(
            (self.num_observations, 128, self.num_actions), self.sequence_length
        )
        self.mlp_critic = CriticMLP(
            (self.num_observations + self.num_actions, 128, 128, 1)
        )
        self.lstm_critic = CriticLSTM(
            (self.num_observations + self.num_actions, 128, 1), self.sequence_length
        )
        self.states = torch.rand(
            (self.num_envs, self.num_observations), device=self.device
        )
        self.lstm_states = torch.rand(
            (self.num_envs, self.sequence_length, self.num_observations),
            device=self.device,
        )
        self.actions = torch.rand((self.num_envs, self.num_actions), device=self.device)

    def test_should_feed_forward_MLP_actor(self):
        means = self.mlp_actor.forward(self.states)
        self.assertIsInstance(means, torch.Tensor)
        self.assertEqual(means.shape, (self.num_envs, self.num_actions))

    def test_should_feed_forward_LSTM_actor(self):
        means = self.lstm_actor.forward(self.lstm_states)
        self.assertIsInstance(means, torch.Tensor)
        self.assertEqual(means.shape, (self.num_envs, self.num_actions))

    def test_should_feed_forward_MLP_critic(self):
        actions = self.mlp_actor.forward(self.states)
        state_action_values = torch.cat((self.states, actions), dim=1)
        values = self.mlp_critic.forward(state_action_values)
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (self.num_envs, 1))

    def test_should_feed_forward_LSTM_critic(self):
        actions = self.lstm_actor.forward(self.lstm_states)
        lstm_actions = actions.unsqueeze(1).repeat(1, self.sequence_length, 1)
        state_action_values = torch.cat((self.lstm_states, lstm_actions), dim=2)
        values = self.lstm_critic.forward(state_action_values)
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (self.num_envs, 1))


if __name__ == "__main__":
    unittest.main()
