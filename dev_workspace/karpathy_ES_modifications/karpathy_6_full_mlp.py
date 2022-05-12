"""
A bare bones examples of optimizing a black-box function using Natural Evolution
Strategies (NES), where the parameter distribution is a gaussian of fixed standard
deviation. -> parallelized with PyTorch and uses a full-on neural network
"""

import numpy as np
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# hyperparameters
num_envs = 512
noise_std_dev = 0.02
learning_rate = 0.01
solution = torch.tensor((0.5, 0.1, -0.3), device=device)
seed = 0

# For the purposes of an example, lets try to minimize the L2 distance to a
# specific solution vector. So the highest reward we can achieve is 0, when our vector
# of weights is exactly equal to the solution.

expanded_solution = solution.unsqueeze(0).unsqueeze(0)
# the black-box function we want to optimize
def black_box_function(actions: torch.Tensor) -> torch.Tensor:
    while len(actions.shape) < 3:
        actions = actions.unsqueeze(0)
    reward = -torch.sum(torch.square(actions - expanded_solution), dim=2)
    return reward


# start the optimization with a random initial guess
torch.manual_seed(seed)
states = torch.randn((1, 24), device=device)
duplicated_states = states.repeat(num_envs, 1, 1)
layers = [
    torch.normal(0, np.sqrt(2 / 24), (24, 10), device=device),
    torch.normal(0, np.sqrt(2 / 24), (1, 10), device=device),
    torch.nn.ELU(),
    torch.normal(0, np.sqrt(2 / 10), (10, 3), device=device),
    torch.normal(0, np.sqrt(2 / 10), (1, 3), device=device),
    torch.nn.Identity(),
]
perturbed_layers: "list[torch.Tensor]" = []


def forward(states: torch.Tensor) -> torch.Tensor:
    computation = states
    while len(computation.shape) < 3:
        computation = computation.unsqueeze(0)
    for idx in range(0, len(layers), 3):
        weights = layers[idx]
        biases = layers[idx + 1]
        activation = layers[idx + 2]
        computation = activation(torch.matmul(computation, weights) + biases)
    actions = computation.squeeze(1)
    return actions


print(f"solution: {str(solution)}")

for i in range(601):
    if i % 30 == 0:
        # fitness of current weights
        actions = forward(states)
        print(f"iter: {i:4d} | R: {black_box_function(actions).item():0.6f}")

    raise Exception("STOP HERE")

    weight_perturbations = torch.randn((num_envs, 24, 3), device=device)
    bias_perturbations = torch.randn((num_envs, 1, 3), device=device)
    duplicated_weights = weights.repeat(num_envs, 1, 1)
    duplicated_biases = biases.repeat(num_envs, 1, 1)
    perturbed_weights = duplicated_weights + noise_std_dev * weight_perturbations
    perturbed_biases = duplicated_biases + noise_std_dev * bias_perturbations

    new_actions = activation(
        torch.matmul(duplicated_states, perturbed_weights) + perturbed_biases
    )

    returns = black_box_function(new_actions).unsqueeze(-1)

    mean_weight_grad = torch.mul(returns, weight_perturbations).mean(0)
    mean_bias_grad = torch.mul(returns, bias_perturbations).mean(0)
    weights += learning_rate / noise_std_dev * mean_weight_grad
    biases += learning_rate / noise_std_dev * mean_bias_grad
