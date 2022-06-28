import isaacgym
import torch
from finenvs.agents.SAC.SAC_agent import SACAgentMLP
from finenvs.environments.isaac_gym_env import IsaacGymEnv
from finenvs.environments.isaac_gym_envs.utils.config_utils import (
    get_isaac_gym_env_args,
)

# FIXME: figure out why Humanoid training breaks down after a long time!
# probably exploding gradients... similar thing happened with PPO
# seed 0: breaks after 15 m XX s.
# seed 1: breaks after 6 m 47 s.
# seed 2: breaks after 7 min 6 s.
# seems that it's more susceptible to breaking with smaller starting_alpha values?


def train_SAC_MLP_on_environiment(env_name: str):
    torch.manual_seed(0)

    env_args = get_isaac_gym_env_args(env_name)
    num_envs = env_args["num_envs"]
    desired_eval_return = 490

    env = IsaacGymEnv(env_name, num_envs, headless=True)
    agent = SACAgentMLP(
        env_args,
        hidden_dims=(256, 256),
        mini_batch_size=num_envs,
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
    train_SAC_MLP_on_environiment("Cartpole")
