import torch
import unittest
from environments.time_series_env import TimeSeriesEnv


class TestTimeSeriesEnv(unittest.TestCase):
    def setUp(self):
        self.IBM_env = TimeSeriesEnv("IBM")

    def test_should_reset_IBM_env(self):
        self.assertIsInstance(self.IBM_env.reset(), torch.Tensor)

    def test_should_step_IBM_env(self):
        self.IBM_env.step()


if __name__ == "__main__":
    unittest.main()
