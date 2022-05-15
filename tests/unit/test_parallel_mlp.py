import numpy as np
import torch
import unittest
from finevo.agents.networks.parallel_mlp import ParallelMLP


class TestParallelMLPNetworks(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.num_envs = 8
        self.num_observations = 24
        self.num_actions = 3
        self.network = ParallelMLP(
            self.num_envs,
            (self.num_observations, 5, 4, self.num_actions),
            noise_std_dev=0.02,
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

    def test_5_should_compute_centered_rank(self):
        values = torch.tensor(range(31, 26, -1), device=self.network.device)
        centered_ranks = ParallelMLP.compute_centered_rank(values)
        true_centered_ranks = torch.tensor(
            np.arange(0.5, -0.75, -0.25), device=self.network.device
        )
        difference = torch.norm(centered_ranks - true_centered_ranks).item()
        self.assertAlmostEqual(difference, 0)


if __name__ == "__main__":
    unittest.main()
