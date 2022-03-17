import torch
import unittest
from environments.time_series_env import TimeSeriesEnv
from typing import Dict, Tuple


class TestTimeSeriesEnv(unittest.TestCase):
    def setUp(self):
        self.num_envs = 4
        self.IBM_env = TimeSeriesEnv("IBM", self.num_envs, testing_code=True)

    def test_should_reset_IBM_env(self):
        self.assertIsInstance(self.IBM_env.reset(), torch.Tensor)

    def test_should_step_IBM_env(self):
        actions = torch.rand((self.num_envs, 1), device=self.IBM_env.device) * 2 - 1
        step_info: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, Dict
        ] = self.IBM_env.step(actions)
        self.assertIsInstance(step_info, tuple)
        (next_states, rewards, dones, info) = step_info
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
