import torch
import unittest
from finevo.environments.time_series_env import TimeSeriesEnv
from typing import Dict, Tuple


class TestSPYTraining(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.num_envs = 4
        self.env = TimeSeriesEnv("SPY", self.num_envs, testing_code=False)

    def test_should_reset_envs(self):
        states = self.env.reset()
        self.assertIsInstance(states, torch.Tensor)

    def test_should_step_envs(self):
        actions = torch.rand((self.num_envs, 1), device=self.env.device) * 2 - 1
        step_info: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, Dict
        ] = self.env.step(actions)
        self.assertIsInstance(step_info, tuple)
        (next_states, rewards, dones, info) = step_info
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(info, dict)

    def test_should_step_dataset(self):
        self.env.step_dataset("train")


if __name__ == "__main__":
    unittest.main()
