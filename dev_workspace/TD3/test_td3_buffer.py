import torch
import unittest
from TD3_buffer import Buffer
from typing import Dict

class TestBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.num_envs = 5
        cls.num_observations = 3
        cls.num_actions = 3
        cls.num_steps = 16
        cls.buffer = Buffer()

    def store_sample_data(self):
        states = (
            torch.rand(
                (self.num_envs, self.num_observations), device=self.buffer.device
            )
            * 2
            - 1
        )
        actions = (
            torch.rand((self.num_envs, 1), device=self.buffer.device) * 2
            - 1
        )
        target_actions = (
            torch.rand((self.num_envs, 1), device=self.buffer.device) * 2
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
        state_action_values_1 = torch.rand((self.num_envs, 1), device=self.buffer.device) * 2 - 1
        state_action_values_2 = torch.rand((self.num_envs, 1), device=self.buffer.device) * 2 - 1
        state_action_values_target_1 = torch.rand((self.num_envs, 1), device=self.buffer.device) * 2 - 1
        state_action_values_target_2 = torch.rand((self.num_envs, 1), device=self.buffer.device) * 2 - 1
        
        self.buffer.store(states, actions, rewards, dones, next_states,target_actions,
            state_action_values_1, state_action_values_2, state_action_values_target_1, 
            state_action_values_target_2)
    
    def test_1_should_store_sample_data_to_buffer(self):
        for _ in range(self.num_steps):
            self.store_sample_data()
    
    def test_2_should_accurately_return_current_size_of_buffer(self):
        self.assertEqual(self.buffer.size(), self.num_envs * self.num_steps)
    
    def test_3_should_reshape_buffer(self):
        size = self.buffer.size()
        self.buffer.prepare_training_data()
        for item in self.buffer.container.values():
            item_length = item.shape[0]
            self.assertEqual(size, item_length)

    def test_4_should_shuffle_buffer(self):
        self.buffer.shuffle()
    
    def test_5_should_retrieve_batches(self):
        batch_dict = self.buffer.get_batches()
        self.assertIsInstance(batch_dict, Dict)
        for key, value in batch_dict.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, torch.Tensor)

    def test_6_should_retrieve_proper_mini_batch_indices(self):
        mini_batch_indices = self.buffer.get_mini_batch_indices()
        self.assertEqual(self.buffer.num_mini_batches, len(mini_batch_indices))


if __name__ == "__main__":
    unittest.main()