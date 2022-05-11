import torch
from finevo.agents.evo_agent import EvoAgent

device = "cuda:0" if torch.cuda.is_available() else "cpu"
solution = torch.tensor((0.5, 0.1, -0.3), device=device)

expanded_solution = solution.unsqueeze(0)


def black_box_function(weights: torch.Tensor) -> torch.Tensor:
    if len(weights.shape) == 1:
        weights.unsqueeze(0)
    reward = -torch.sum(torch.square(weights - expanded_solution), dim=1)
    return reward


def train_Evo_MLP_on_black_box():
    num_envs = 512
    env_args = {
        "env_name": "black_box",
        "num_envs": num_envs,
        "num_observations": 1,
        "num_actions": 3,
    }
    max_training_steps = 2

    torch.manual_seed(0)
    states = torch.randn((1,), device=device).repeat(num_envs, 1)
    agent = EvoAgent(env_args, write_to_csv=False)

    training_steps = 0
    while training_steps < max_training_steps:
        actions = agent.step(states)
        print(actions)
        rewards = black_box_function(actions)
        dones = torch.ones((num_envs,), device=device)
        agent.store(rewards, dones)
        agent.train()
        training_steps += 1


if __name__ == "__main__":
    train_Evo_MLP_on_black_box()
