import torch
from finenvs.agents.TD3.TD3_agent import TD3AgentMLP
from finenvs.environments.time_series_env import TimeSeriesEnv


def train_TD3_MLP_on_training_set(env_name: str, seed: int):
    torch.manual_seed(seed)
    num_envs = 4096
    env = TimeSeriesEnv(
        instrument_name=env_name,
        dataset_key="train",
        num_envs=num_envs,
    )
    max_samples = 100_000_000_000
    env_args = env.get_env_args()
    agent = TD3AgentMLP(
        env_args=env_args,
        hidden_dims=(1024, 512, 256, 128),
        mini_batch_size=25,
        model_save_interval=400,
        write_to_csv=True,
    )

    states = env.reset()
    total_samples = 0
    while total_samples < max_samples:
        actions = agent.step(states)
        (next_states, rewards, dones, _) = env.step(actions)
        agent.store(states, actions, rewards, next_states, dones)
        states = next_states
        (total_samples, _) = agent.train()


if __name__ == "__main__":
    train_TD3_MLP_on_training_set("SPY", 602585557)
