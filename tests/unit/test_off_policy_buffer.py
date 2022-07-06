import torch
import unittest
from finenvs.agents.off_policy_buffer import Buffer
from typing import Dict


class TestBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.num_envs = 512
        self.max_size = 10_000
        self.num_observations = 24
        self.num_actions = 3
        self.num_steps = 16
        self.buffer = Buffer(self.max_size)

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
        next_states = (
            torch.rand(
                (self.num_envs, self.num_observations), device=self.buffer.device
            )
            * 2
            - 1
        )
        self.buffer.store(states, actions, rewards, next_states, dones)

    def test_1_should_store_sample_data_to_buffer(self):
        for _ in range(self.num_steps):
            self.store_sample_data()

    def test_2_should_accurately_return_current_size_of_buffer(self):
        self.assertEqual(self.buffer.size(), self.num_envs * self.num_steps)

    def test_3_should_max_out_buffer(self):
        while self.buffer.size() < self.max_size:
            self.store_sample_data()
        self.assertEqual(self.buffer.size(), self.max_size)
        for _ in range(self.num_steps):
            self.store_sample_data()
        self.assertEqual(self.buffer.size(), self.max_size)

    def test_4_should_retrieve_mini_batch(self):
        mini_batch_dict = self.buffer.get_mini_batch(100)
        self.assertIsInstance(mini_batch_dict, Dict)
        for key, value in mini_batch_dict.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, torch.Tensor)
            self.assertEqual(value.shape[0], 100)


if __name__ == "__main__":
    unittest.main()
