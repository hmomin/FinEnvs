import isaacgym
from finenvs.agents.TD3.TD3_agent import TD3AgentLSTM
from finenvs.environments.isaac_gym_env import IsaacGymEnv
from finenvs.environments.isaac_gym_envs.utils.config_utils import (
    get_isaac_gym_env_args,
)


def train_TD3_LSTM_on_environiment(env_name: str):
    env_args = get_isaac_gym_env_args(env_name)
    num_envs = env_args["num_envs"]
    sequence_length = env_args["sequence_length"]
    desired_eval_return = 490

    env = IsaacGymEnv(env_name, num_envs, headless=True)
    agent = TD3AgentLSTM(
        env_args, hidden_dim=1024, model_save_interval=-1, write_to_csv=False
    )
    states = env.reset()
    states = states.unsqueeze(1).repeat(1, sequence_length, 1)
    while True:
        actions = agent.step(states)
        (next_states, rewards, dones, _) = env.step(actions)
        next_states = next_states.unsqueeze(1).repeat(1, sequence_length, 1)
        agent.store(states, actions, rewards, next_states, dones)
        states = next_states
        (_, evaluation_return) = agent.train()
        if evaluation_return >= desired_eval_return:
            break


if __name__ == "__main__":
    train_TD3_LSTM_on_environiment("Cartpole")
