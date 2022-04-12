import torch
import unittest
from finevo.agents.buffer import Buffer
from typing import Dict


class TestBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.num_envs = 64
        self.num_observations = 24
        self.num_actions = 3
        self.num_steps = 4
        self.buffer = Buffer()

    def store_sample_data(self):
        states = (
            torch.rand(
                (self.num_envs, self.num_observations), device=self.buffer.device
            )
            * 2
            - 1
        )
        actions = (
            torch.rand((self.num_envs, self.num_actions), device=self.buffer.device) * 2
            - 1
        )
        rewards = torch.ones((self.num_envs,), device=self.buffer.device)
        dones = torch.randint(0, 2, (self.num_envs,), device=self.buffer.device)
        log_probs = (
            torch.rand((self.num_envs, self.num_actions), device=self.buffer.device) - 1
        )
        values = torch.rand((self.num_envs, 1), device=self.buffer.device) * 2 - 1
        self.buffer.store(states, actions, rewards, dones, log_probs, values)

    def test_1_should_store_sample_data_to_buffer(self):
        for _ in range(self.num_steps):
            self.store_sample_data()

    def test_2_should_accurately_return_current_size_of_buffer(self):
        self.assertEqual(self.buffer.size(), self.num_envs * self.num_steps)

    def test_3_should_compute_returns_and_advantages(self):
        last_values = torch.rand((self.num_envs, 1), device=self.buffer.device) * 2 - 1
        self.buffer.compute_returns_and_advantages(last_values)
        rewards = self.buffer.container["rewards"]
        returns = self.buffer.container["returns"]
        advantages = self.buffer.container["advantages"]
        self.assertEqual(returns.shape, rewards.shape)
        self.assertEqual(advantages.shape, rewards.shape)

    def test_4_should_reshape_buffer(self):
        size = self.buffer.size()
        self.buffer.reshape()
        for item in self.buffer.container.values():
            item_length = item.shape[0]
            self.assertEqual(size, item_length)

    def test_5_should_shuffle_buffer(self):
        self.buffer.shuffle()

    def test_6_should_retrieve_batches(self):
        batch_dict = self.buffer.get_batches()
        self.assertIsInstance(batch_dict, Dict)
        for key, value in batch_dict.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, torch.Tensor)

    def test_7_should_retrieve_proper_mini_batch_indices(self):
        mini_batch_indices = self.buffer.get_mini_batch_indices()
        self.assertEqual(self.buffer.num_mini_batches, len(mini_batch_indices))


if __name__ == "__main__":
    unittest.main()
