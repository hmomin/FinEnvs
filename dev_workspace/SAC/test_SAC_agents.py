import isaacgym
import unittest
from agent import SACAgentLSTM, SACAgentMLP
from finenvs.environments.isaac_gym_env import IsaacGymEnv
from finenvs.environments.isaac_gym_envs.utils.config_utils import (
    get_isaac_gym_env_args,
)


class TestSACAgents(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env_name = "Cartpole"
        cls.env_args = get_isaac_gym_env_args(cls.env_name)
        cls.num_envs = cls.env_args["num_envs"]
        cls.num_steps = 3
        cls.batch_size = cls.num_envs
        cls.env = IsaacGymEnv(cls.env_name, cls.num_envs, headless=True)
        cls.mlp_agent = SACAgentMLP(cls.env_args, write_to_csv=False)
        cls.lstm_agent = SACAgentLSTM(cls.env_args, write_to_csv=False)

    def test_should_step_mlp_environiment_and_train(self):
        states = self.env.reset()
        for _ in range(self.num_steps):
            actions = self.mlp_agent.step(states)
            (next_states, rewards, dones, _) = self.env.step(actions)
            self.mlp_agent.store(states, actions, rewards, next_states, dones)
            states = next_states
            self.mlp_agent.train()

    def test_should_step_lstm_environiment_and_train(self):
        states = self.env.reset()
        for _ in range(self.num_steps):
            actions = self.lstm_agent.step(states)
            (next_states, rewards, dones, _) = self.env.step(actions)
            self.lstm_agent.store(states, actions, rewards, next_states, dones)
            states = next_states
            self.lstm_agent.train()


if __name__ == "__main__":
    unittest.main()