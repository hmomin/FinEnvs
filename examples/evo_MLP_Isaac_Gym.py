import isaacgym
import torch
from finevo.agents.evo_agent import EvoAgent
from finevo.environments.isaac_gym_env import IsaacGymEnv
from finevo.environments.isaac_gym_envs.utils.config_utils import (
    get_isaac_gym_env_args,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(42)


def train_Evo_MLP_on_environment(env_name: str):
    num_envs = 4096 + 1
    episodes_per_batch = 10_000
    timesteps_per_batch = 500_000

    env_args = get_isaac_gym_env_args(env_name)
    env_args["num_envs"] = num_envs

    env = IsaacGymEnv(env_name, num_envs, headless=False)
    agent = EvoAgent(
        env_args,
        hidden_dims=(256, 256),
        learning_rate=0.01,
        noise_std_dev=0.02,
        l2_coefficient=0.005,
        write_to_csv=False,
    )
    states = env.reset()
    steps = 0
    while True:
        actions = agent.step(states)
        (next_states, rewards, dones, _) = env.step(actions)
        num_done = agent.store(rewards, dones)
        states = next_states
        steps += num_envs
        if num_done >= episodes_per_batch or steps >= timesteps_per_batch:
            agent.train()
            states = env.reset_all()
            steps = 0


if __name__ == "__main__":
    train_Evo_MLP_on_environment("Humanoid")
