import isaacgym
from finenvs.agents.SAC.SAC_agent import SACAgentMLP
from finenvs.environments.isaac_gym_env import IsaacGymEnv
from finenvs.environments.isaac_gym_envs.utils.config_utils import (
    get_isaac_gym_env_args,
)


def train_SAC_MLP_on_environiment(env_name: str):
    env_args = get_isaac_gym_env_args(env_name)
    num_envs = env_args["num_envs"]
    max_samples = 1_000_000_000
    desired_eval_return = 490

    env = IsaacGymEnv(env_name, num_envs, headless=True)
    agent = SACAgentMLP(env_args, hidden_dims=(256, 256), write_to_csv=False)
    states = env.reset()
    total_samples = 0
    while total_samples < max_samples:
        actions = agent.step(states)
        (next_states, rewards, dones, _) = env.step(actions)
        agent.store(states, actions, rewards, next_states, dones)
        states = next_states
        (total_samples, evaluation_return) = agent.train()
        if evaluation_return >= desired_eval_return:
            break


if __name__ == "__main__":
    train_SAC_MLP_on_environiment("Cartpole")
