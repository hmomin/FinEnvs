import torch
import unittest
from finevo.agents.networks.parallel_mlp import ParallelMLP


class TestParallelMLPNetworks(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.num_envs = 32
        self.num_observations = 24
        self.num_actions = 3
        self.network = ParallelMLP(
            self.num_envs, (self.num_observations, 512, 256, self.num_actions)
        )
        self.states = torch.randn(
            (self.num_envs, self.num_observations), device=self.network.device
        )
        self.fitnesses = torch.randn((self.num_envs,), device=self.network.device)

    def test_1_should_perturb_parameters(self):
        self.network.perturb_parameters()

    def test_2_should_feed_forward_network(self):
        outputs = self.network.forward(self.states)
        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.shape, (self.num_envs, self.num_actions))

    def test_3_should_reconstruct_perturbations(self):
        self.network.reconstruct_perturbations()

    def test_4_should_update_parameters(self):
        self.network.update_parameters(self.fitnesses)


if __name__ == "__main__":
    unittest.main()
