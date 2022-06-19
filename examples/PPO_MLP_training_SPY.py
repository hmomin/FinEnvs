import torch
from finenvs.agents.PPO.PPO_agent import PPOAgentMLP
from finenvs.environments.time_series_env import TimeSeriesEnv


def train_PPO_MLP_on_training_set(env_name: str):
    torch.manual_seed(292408520)
    num_envs = 4096
    env = TimeSeriesEnv(
        instrument_name=env_name,
        dataset_key="train",
        num_envs=num_envs,
    )
    batch_size = num_envs * 32
    max_samples = 100_000_000_000
    env_args = env.get_env_args()
    agent = PPOAgentMLP(
        env_args=env_args,
        num_epochs=4,
        num_mini_batches=4,
        hidden_dims=(1024, 512, 256, 128),
        model_save_interval=20,
        write_to_csv=True,
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
    train_PPO_MLP_on_training_set("SPY")
