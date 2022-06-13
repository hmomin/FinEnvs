import torch
import unittest
from finenvs.agents.networks.multilayer_perceptron import MLPNetwork


class TestMLPNetworks(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.num_envs = 32
        self.num_observations = 24
        self.num_actions = 3
        self.network = MLPNetwork((self.num_observations, 512, 256, self.num_actions))
        self.states = torch.rand(
            (self.num_envs, self.num_observations), device=self.network.device
        )

    def test_should_feed_forward_network(self):
        outputs = self.network.forward(self.states)
        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.shape, (self.num_envs, self.num_actions))


if __name__ == "__main__":
    unittest.main()
