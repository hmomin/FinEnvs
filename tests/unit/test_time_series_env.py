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
        step_info: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, Dict
        ] = self.IBM_env.step(
            torch.rand((self.num_envs,), device=self.IBM_env.device) * 2 - 1
        )


if __name__ == "__main__":
    unittest.main()
