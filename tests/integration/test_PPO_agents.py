import isaacgym
import unittest
from finevo.agents.PPO_agent import PPOAgentMLP, PPOAgentLSTM
from finevo.environments.isaac_gym_env import IsaacGymEnv
from finevo.environments.isaac_gym_envs.utils.config_utils import (
    get_isaac_gym_env_args,
)


class TestPPOAgents(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_name = "BallBalance"
        self.env_args = get_isaac_gym_env_args(self.env_name)
        self.num_envs = self.env_args["num_envs"]
        self.num_steps = 3
        self.batch_size = self.num_envs

        self.env = IsaacGymEnv(self.env_name, self.num_envs, headless=True)
        self.mlp_agent = PPOAgentMLP(self.env_args, write_to_csv=False)
        self.lstm_agent = PPOAgentLSTM(self.env_args, write_to_csv=False)

    def test_should_step_MLP_agent_through_the_environment_and_train(self):
        states = self.env.reset()
        for _ in range(self.num_steps):
            (actions, log_probs, values) = self.mlp_agent.step(states)
            (next_states, rewards, dones, _) = self.env.step(actions)
            self.mlp_agent.store(states, actions, rewards, dones, log_probs, values)
            states = next_states
            if self.mlp_agent.get_buffer_size() >= self.batch_size:
                self.mlp_agent.train(states)

    def test_should_step_LSTM_agent_through_the_environment_and_train(self):
        states = self.env.reset()
        for _ in range(self.num_steps):
            (actions, log_probs, values) = self.lstm_agent.step(states)
            (next_states, rewards, dones, _) = self.env.step(actions)
            self.lstm_agent.store(states, actions, rewards, dones, log_probs, values)
            states = next_states
            if self.lstm_agent.get_buffer_size() >= self.batch_size:
                self.lstm_agent.train(states)


if __name__ == "__main__":
    unittest.main()
