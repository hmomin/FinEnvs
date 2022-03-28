import isaacgym
import unittest
from finevol.agents.PPO_agent_MLP import PPOAgentMLP
from finevol.environments.isaac_gym_env import IsaacGymEnv
from finevol.environments.isaac_gym_envs.utils.config_utils import (
    get_isaac_gym_env_args,
)


class TestPPOAgentMLP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_name = "BallBalance"
        self.env_args = get_isaac_gym_env_args(self.env_name)
        self.num_envs = self.env_args["num_envs"]
        self.num_steps = 3
        self.batch_size = self.num_envs

        self.env = IsaacGymEnv(self.env_name, self.num_envs, headless=True)
        self.agent = PPOAgentMLP(self.env_args, write_to_csv=False)

    def test_should_step_the_agent_through_the_environment_and_train(self):
        states = self.env.reset()
        for _ in range(self.num_steps):
            (actions, log_probs, values) = self.agent.step(states)
            (next_states, rewards, dones, _) = self.env.step(actions)
            self.agent.store(states, actions, rewards, dones, log_probs, values)
            states = next_states
            if self.agent.get_buffer_size() >= self.batch_size:
                self.agent.train(states)


if __name__ == "__main__":
    unittest.main()
