import torch
import unittest
from finevo.agents.continuous_actor import (
    ContinuousActorMLP,
    ContinuousActorLSTM,
)
from finevo.agents.critic import CriticMLP, CriticLSTM
from finevo.device_utils import set_device


class TestActorCritic(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.num_envs = 32
        self.num_observations = 24
        self.num_actions = 3
        self.device = set_device(0)
        self.mlp_actor = ContinuousActorMLP(
            (self.num_observations, 128, 128, self.num_actions),
        )
        self.lstm_actor = ContinuousActorLSTM(
            (self.num_observations, 128, self.num_actions),
        )
        self.mlp_critic = CriticMLP((self.num_observations, 128, 128, 1))
        self.lstm_critic = CriticLSTM((self.num_observations, 128, 1))
        self.states = torch.rand(
            (self.num_envs, self.num_observations), device=self.device
        )

    def test_should_feed_forward_MLP_actor(self):
        means = self.mlp_actor.forward(self.states)
        self.assertIsInstance(means, torch.Tensor)
        self.assertEqual(means.shape, (self.num_envs, self.num_actions))

    def test_should_feed_forward_LSTM_actor(self):
        means = self.lstm_actor.forward(self.states)
        self.assertIsInstance(means, torch.Tensor)
        self.assertEqual(means.shape, (self.num_envs, self.num_actions))

    def test_should_feed_forward_MLP_critic(self):
        values = self.mlp_critic.forward(self.states)
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (self.num_envs, 1))

    def test_should_feed_forward_LSTM_critic(self):
        values = self.lstm_critic.forward(self.states)
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (self.num_envs, 1))

    def test_should_get_MLP_log_probs_of_actions(self):
        distribution = self.mlp_actor.get_distribution(self.states)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (self.num_envs, self.num_actions))

    def test_should_get_LSTM_log_probs_of_actions(self):
        distribution = self.lstm_actor.get_distribution(self.states)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (self.num_envs, self.num_actions))


if __name__ == "__main__":
    unittest.main()
