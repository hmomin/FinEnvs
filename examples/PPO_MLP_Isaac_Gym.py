import isaacgym
from finevo.agents.PPO_agent import PPOAgentMLP
from finevo.environments.isaac_gym_env import IsaacGymEnv
from finevo.environments.isaac_gym_envs.utils.config_utils import get_isaac_gym_env_args


def train_PPO_MLP_on_environment(env_name: str):
    env_args = get_isaac_gym_env_args(env_name)
    num_envs = env_args["num_envs"]
    batch_size = num_envs * 32
    max_samples = 1_000_000_000

    env = IsaacGymEnv(env_name, num_envs, headless=True)
    agent = PPOAgentMLP(env_args, hidden_dims=(400, 200, 100), write_to_csv=True)
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
    train_PPO_MLP_on_environment("Humanoid")
