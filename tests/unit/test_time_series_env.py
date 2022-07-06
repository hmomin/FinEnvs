import torch
import unittest
from finenvs.environments.time_series_env import TimeSeriesEnv
from typing import Dict, Tuple


class TestTimeSeriesEnv(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.envs = [
            TimeSeriesEnv("IBM", "dummy"),
            TimeSeriesEnv("OIH", "dummy"),
            TimeSeriesEnv("SPY", "dummy"),
        ]

    def test_should_reset_envs(self):
        for env in self.envs:
            self.assertIsInstance(env.reset(), torch.Tensor)

    def test_should_step_envs(self):
        for env in self.envs:
            self.step_helper(env)

    def step_helper(self, env: TimeSeriesEnv):
        num_envs = env.num_envs
        actions = torch.rand((num_envs, 1), device=env.device) * 2 - 1
        step_info: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict] = env.step(
            actions
        )
        self.assertIsInstance(step_info, tuple)
        (next_states, rewards, dones, info) = step_info
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(info, dict)

    def test_should_step_envs_1000_times(self):
        for _ in range(1000):
            for env in self.envs:
                self.step_helper(env)


if __name__ == "__main__":
    unittest.main()
