import torch
import unittest
from TD3_Critic import TD3CriticMLP, TD3CriticLSTM
from TD3_actor import TD3ActorMLP, TD3ActorLSTM
from finevo.device_utils import set_device


class TestActorCritic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_envs = 32
        cls.num_observations = 24
        cls.num_actions = 3
        cls.device = set_device(0)
        cls.mlp_actor = TD3ActorMLP((cls.num_observations, 128, 128, 1))
        cls.lstm_actor = TD3ActorLSTM((cls.num_observations, 128, 1))
        cls.mlp_critic = TD3CriticMLP((cls.num_observations + 1, 128, 128, 1))
        cls.lstm_critic = TD3CriticLSTM((cls.num_observations + 1, 128, 1))
        cls.states = torch.rand((cls.num_envs, cls.num_observations), device=cls.device)
        cls.actions = torch.rand(
            (cls.num_envs, cls.num_observations), device=cls.device
        )

    def test_should_feed_forward_MLP_actor(self):
        means = self.mlp_actor.forward(self.states)
        self.assertIsInstance(means, torch.Tensor)
        self.assertEqual(means.shape, (self.num_envs, 1))

    def test_should_feed_forward_LSTM_actor(self):
        means = self.lstm_actor.forward(self.states)
        self.assertIsInstance(means, torch.Tensor)
        self.assertEqual(means.shape, (self.num_envs, 1))

    def test_should_feed_forward_MLP_critic(self):
        x = self.mlp_actor.forward(self.states)
        y = torch.cat((self.states, x), dim=1)
        values = self.mlp_critic.forward(y)
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (self.num_envs, 1))

    def test_should_feed_forward_LSTM_critic(self):
        x = self.mlp_actor.forward(self.states)
        y = torch.cat((self.states, x), dim=1)
        values = self.lstm_critic.forward(y)
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (self.num_envs, 1))


if __name__ == "__main__":
    unittest.main()
