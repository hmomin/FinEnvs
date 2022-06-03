import isaacgym
import unittest
from TD3_agent import TD3AgentLSTM, TD3AgentMLP
from finevo.environments.isaac_gym_env import IsaacGymEnv
from finevo.environments.isaac_gym_envs.utils.config_utils import (
    get_isaac_gym_env_args,
)

print(get_isaac_gym_env_args("BallBalance"))


class TestTD3Agents(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env_name = "BallBalance"
        cls.env_args = get_isaac_gym_env_args(cls.env_name)
        cls.num_envs = cls.env_args["num_envs"]
        cls.num_steps = 3
        cls.batch_size = cls.num_envs
        cls.env = IsaacGymEnv(cls.env_name, cls.num_envs, headless=True)
        cls.mlp_agent = TD3AgentMLP(cls.env_args, write_to_csv=False)
        cls.lstm_agent = TD3AgentLSTM(cls.env_args, write_to_csv=False)

    def test_should_step_mlp_environiment_and_train(self):
        states = self.env.reset()
        for _ in range(self.num_steps):
            actions, sa_values_1, sa_values_2 = self.mlp_agent.step(states)
            (next_states, rewards, dones, _) = self.env.step(actions)
            (
                target_actions,
                target_sa_values_1,
                target_sa_values_2,
            ) = self.mlp_agent.target_networks_step(next_states)
            states = next_states
            self.mlp_agent.store(
                states,
                actions,
                rewards,
                dones,
                next_states,
                target_actions,
                sa_values_1,
                sa_values_2,
                target_sa_values_1,
                target_sa_values_2,
            )
            if self.mlp_agent.get_buffer_size() >= self.batch_size:
                self.mlp_agent.train(states)

    def test_should_step_lstm_environiment_and_train(self):
        states = self.env.reset()
        for _ in range(self.num_steps):
            actions, sa_values_1, sa_values_2 = self.mlp_agent.step(states)
            (next_states, rewards, dones, _) = self.env.step(actions)
            (
                target_actions,
                target_sa_values_1,
                target_sa_values_2,
            ) = self.mlp_agent.target_networks_step(next_states)
            states = next_states
            self.lstm_agent.store(
                states,
                actions,
                rewards,
                dones,
                next_states,
                target_actions,
                sa_values_1,
                sa_values_2,
                target_sa_values_1,
                target_sa_values_2,
            )
            if self.lstm_agent.get_buffer_size() >= self.batch_size:
                self.lstm_agent.train(states)


if __name__ == "__main__":
    unittest.main()
