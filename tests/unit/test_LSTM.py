import torch
import torch.nn as nn
import unittest
from finevo.agents.networks.lstm import LSTMNetwork


class TestLSTMNetworks(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.num_envs = 32
        self.num_observations = 24
        self.num_actions = 3
        self.lstm = LSTMNetwork((self.num_observations, 137, self.num_actions), nn.Tanh)
        self.states = torch.rand(
            (self.num_envs, self.num_observations), device=self.lstm.device
        )

    def test_should_not_instantiate_lstm_with_more_than_3_layers(self):
        self.assertRaises(Exception, LSTMNetwork, (1, 2, 3, 4))

    def test_should_feed_forward_lstm(self):
        outputs = self.lstm.forward(self.states)
        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.shape, (self.num_envs, self.num_actions))


if __name__ == "__main__":
    unittest.main()
