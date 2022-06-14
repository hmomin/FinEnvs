import csv
import isaacgym
import torch
from finenvs.environments.isaac_gym_env import IsaacGymEnv
from finenvs.environments.isaac_gym_envs.utils.config_utils import (
    get_isaac_gym_env_args,
)
from time import time

torch.manual_seed(42)


def step_isaac_gym_environment(env_name: str):
    num_envs = 2**14
    env_args = get_isaac_gym_env_args(env_name)
    env_args["num_envs"] = num_envs
    actions = torch.randn(
        env_args["num_envs"], env_args["num_actions"], device="cuda:0"
    )
    env = IsaacGymEnv(env_name, num_envs, headless=True)
    states = env.reset()
    for _ in range(1000):
        start = time()
        (next_states, rewards, dones, _) = env.step(actions)
        time_step = [time() - start]
        csv_name = f"{env_name}_{num_envs}.csv"
        with open(csv_name, "a") as f:
            writer = csv.writer(f)
            writer.writerow(time_step)


if __name__ == "__main__":
    step_isaac_gym_environment("Humanoid")
