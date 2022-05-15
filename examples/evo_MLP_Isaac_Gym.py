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
    num_envs = 32
    step_limit = 10
    step_increment = 10
    increment_threshold = 0.50

    env_args = get_isaac_gym_env_args(env_name)
    env_args["num_envs"] = num_envs
    # num_envs = env_args["num_envs"]

    env = IsaacGymEnv(env_name, num_envs, headless=True)
    agent = EvoAgent(
        env_args,
        learning_rate=0.01,
        noise_std_dev=0.02,
        hidden_dims=(256, 256),
        write_to_csv=True,
    )
    states = env.reset()
    steps = 0
    while True:
        actions = agent.step(states)
        (next_states, rewards, dones, _) = env.step(actions)
        num_done = agent.store(rewards, dones)
        states = next_states
        steps += 1
        if num_done == num_envs or steps == step_limit:
            agent.train()
            states = env.reset_all()
            if num_done / num_envs < increment_threshold:
                step_limit += step_increment
                print(f"new step limit: {step_limit}...")
            steps = 0


if __name__ == "__main__":
    train_Evo_MLP_on_environment("Humanoid")
