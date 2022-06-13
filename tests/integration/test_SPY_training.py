import torch
import unittest
from finenvs.environments.time_series_env import TimeSeriesEnv
from typing import Dict, Tuple


class TestSPYTraining(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.num_envs = 1024
        self.train_env = TimeSeriesEnv("SPY", "train", self.num_envs)
        # self.valid_env = TimeSeriesEnv("SPY", "valid")
        # self.test_env = TimeSeriesEnv("SPY", "test")

    def test_1_should_reset_train_env(self):
        states = self.train_env.reset()
        self.assertIsInstance(states, torch.Tensor)

    def test_2_should_step_envs(self):
        self.step_helper(self.train_env)

    def step_helper(self, env: TimeSeriesEnv):
        actions = torch.rand((self.num_envs, 1), device=env.device) * 2 - 1
        step_info: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict] = env.step(
            actions
        )
        self.assertIsInstance(step_info, tuple)
        (next_states, rewards, dones, info) = step_info
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(info, dict)

    def test_3_should_step_envs_1000_times(self):
        for _ in range(1000):
            self.step_helper(self.train_env)


if __name__ == "__main__":
    unittest.main()
