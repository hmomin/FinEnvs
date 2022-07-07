import isaacgym
import torch
from finenvs.agents.TD3.TD3_agent import TD3AgentMLP
from finenvs.environments.isaac_gym_env import IsaacGymEnv
from finenvs.environments.isaac_gym_envs.utils.config_utils import (
    get_isaac_gym_env_args,
)


def train_TD3_MLP_on_environiment(env_name: str):
    torch.manual_seed(0)

    env_args = get_isaac_gym_env_args(env_name)
    num_envs = env_args["num_envs"]
    desired_eval_return = 6_000

    env = IsaacGymEnv(env_name, num_envs, headless=True)
    agent = TD3AgentMLP(
        env_args,
        hidden_dims=(256, 256),
        mini_batch_size=num_envs,
        model_save_interval=-1,
        write_to_csv=False,
    )
    states = env.reset_all()
    while True:
        actions = agent.step(states)
        (next_states, rewards, dones, _) = env.step(actions)
        agent.store(states, actions, rewards, next_states, dones)
        states = next_states
        if agent.get_buffer_size() == agent.buffer.max_size:
            (_, evaluation_return) = agent.train()
            if evaluation_return >= desired_eval_return:
                break


if __name__ == "__main__":
    train_TD3_MLP_on_environiment("Ant")
