import isaacgym
import unittest
from finenvs.agents.TD3.TD3_agent import TD3AgentLSTM, TD3AgentMLP
from finenvs.environments.isaac_gym_env import IsaacGymEnv
from finenvs.environments.isaac_gym_envs.utils.config_utils import (
    get_isaac_gym_env_args,
)


class TestTD3Agents(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.env_name = "BallBalance"
        self.env_args = get_isaac_gym_env_args(self.env_name)
        self.sequence_length = self.env_args["sequence_length"]
        self.num_envs = self.env_args["num_envs"]
        self.num_steps = 3
        self.batch_size = self.num_envs * 2
        self.env = IsaacGymEnv(self.env_name, self.num_envs, headless=True)
        self.mlp_agent = TD3AgentMLP(self.env_args, write_to_csv=False)
        self.lstm_agent = TD3AgentLSTM(self.env_args, write_to_csv=False)

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
        states = states.unsqueeze(1).repeat(1, self.sequence_length, 1)
        for _ in range(self.num_steps):
            actions = self.lstm_agent.step(states)
            (next_states, rewards, dones, _) = self.env.step(actions)
            next_states = next_states.unsqueeze(1).repeat(1, self.sequence_length, 1)
            self.lstm_agent.store(states, actions, rewards, next_states, dones)
            states = next_states
            self.lstm_agent.train()


if __name__ == "__main__":
    unittest.main()
