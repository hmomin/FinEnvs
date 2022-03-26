import torch
import unittest
from finevol.agents.multilayer_perceptron import ContinuousActor, Critic


class TestMLPNetworks(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.num_envs = 32
        self.num_observations = 24
        self.num_actions = 3
        self.actor = ContinuousActor(
            (self.num_observations, 128, 128, self.num_actions),
        )
        self.critic = Critic((self.num_observations, 128, 128, 1))
        self.states = torch.rand(
            (self.num_envs, self.num_observations), device=self.actor.device
        )

    def test_should_feed_forward_actor(self):
        means = self.actor.forward(self.states)
        self.assertIsInstance(means, torch.Tensor)
        self.assertEqual(means.shape, (self.num_envs, self.num_actions))

    def test_should_feed_forward_critic(self):
        values = self.critic.forward(self.states)
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (self.num_envs, 1))

    def test_should_get_log_probs_of_actions(self):
        distribution = self.actor.get_distribution(self.states)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (self.num_envs, self.num_actions))


if __name__ == "__main__":
    unittest.main()
