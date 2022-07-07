import isaacgym
from finenvs.agents.PPO.PPO_agent import PPOAgentMLP
from finenvs.environments.isaac_gym_env import IsaacGymEnv
from finenvs.environments.isaac_gym_envs.utils.config_utils import (
    get_isaac_gym_env_args,
)


def train_PPO_MLP_on_environment(env_name: str):
    env_args = get_isaac_gym_env_args(env_name)
    num_envs = env_args["num_envs"]
    batch_size = num_envs * 16
    max_samples = 1_000_000_000

    env = IsaacGymEnv(env_name, num_envs, headless=True)
    agent = PPOAgentMLP(
        env_args,
        num_epochs=4,
        num_mini_batches=4,
        hidden_dims=(256, 256),
        model_save_interval=-1,
        write_to_csv=False,
    )
    states = env.reset()
    total_samples = 0
    while total_samples < max_samples:
        (actions, log_probs, values) = agent.step(states)
        (next_states, rewards, dones, _) = env.step(actions)
        agent.store(states, actions, rewards, dones, log_probs, values)
        states = next_states
        if agent.get_buffer_size() >= batch_size:
            total_samples = agent.train(states)


if __name__ == "__main__":
    train_PPO_MLP_on_environment("Ant")
