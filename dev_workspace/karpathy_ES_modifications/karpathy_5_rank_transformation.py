"""
A bare bones examples of optimizing a black-box function using Natural Evolution
Strategies (NES), where the parameter distribution is a gaussian of fixed standard
deviation. -> parallelized with PyTorch and uses a full-on hidden layer with mirrored
sampling and a rank transformation
"""

import numpy as np
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# hyperparameters
num_envs = 8192
noise_std_dev = 0.02
learning_rate = 0.0003
solution = torch.tensor((0.5, 0.1, -0.3), device=device)
seed = 0

# But, for the purposes of an example, lets try to minimize the L2 distance to a
# specific solution vector. So the highest reward we can achieve is 0, when our vector
# of weights is exactly equal to the solution.

expanded_solution = solution.unsqueeze(0).unsqueeze(0)
# the black-box function we want to optimize
def black_box_function(actions: torch.Tensor) -> torch.Tensor:
    while len(actions.shape) < 3:
        actions = actions.unsqueeze(0)
    reward = -torch.sum(torch.square(actions - expanded_solution), dim=2)
    return reward


def compute_ranks(values: torch.Tensor):
    values_1d = values.squeeze()
    ranks = torch.empty(values_1d.shape[0], device=device)
    sort_indices = values_1d.argsort()
    linear_ranks = torch.arange(0, values.shape[0], dtype=torch.float32, device=device)
    ranks[sort_indices] = linear_ranks
    ranks = ranks.reshape(values.shape)
    return ranks


def compute_centered_rank(values: torch.Tensor) -> torch.Tensor:
    ranks = compute_ranks(values)
    N = ranks.shape[0]
    centered_ranks = ranks / (N - 1) - 0.5
    return centered_ranks


# start the optimization with a random initial guess
torch.manual_seed(seed)
states = torch.randn((24,), device=device)
duplicated_states = states.repeat(num_envs, 1, 1)

weights = torch.normal(0, np.sqrt(2 / 24), (24, 3), device=device)
biases = torch.normal(0, np.sqrt(2 / 24), (1, 3), device=device)
activation = torch.nn.Tanh()

print(f"solution: {str(solution)}")

for i in range(31):
    if i % 1 == 0:
        # fitness of current weights
        actions = activation(torch.matmul(states, weights) + biases)
        # print(actions)
        print(f"iter: {i:4d} | R: {black_box_function(actions).item():0.6f}")

    positive_weight_perturbations = torch.randn((num_envs // 2, 24, 3), device=device)
    negative_weight_perturbations = -positive_weight_perturbations
    positive_bias_perturbations = torch.randn((num_envs // 2, 1, 3), device=device)
    negative_bias_perturbations = -positive_bias_perturbations

    weight_perturbations = torch.cat(
        [positive_weight_perturbations, negative_weight_perturbations], dim=0
    )
    bias_perturbations = torch.cat(
        [positive_bias_perturbations, negative_bias_perturbations], dim=0
    )

    duplicated_weights = weights.repeat(num_envs, 1, 1)
    duplicated_biases = biases.repeat(num_envs, 1, 1)
    perturbed_weights = duplicated_weights + noise_std_dev * weight_perturbations
    perturbed_biases = duplicated_biases + noise_std_dev * bias_perturbations

    new_actions = activation(
        torch.matmul(duplicated_states, perturbed_weights) + perturbed_biases
    )

    returns = black_box_function(new_actions).unsqueeze(-1)

    (positive_returns, negative_returns) = returns.split(num_envs // 2)
    new_returns = positive_returns - negative_returns

    # print(new_returns)
    # print(max(new_returns), min(new_returns))
    new_returns = compute_centered_rank(new_returns)
    # histogram_data = new_returns.squeeze().cpu().numpy()
    # plt.hist(histogram_data)
    # plt.show()
    # print(new_returns)
    # print(max(new_returns), min(new_returns))

    # raise Exception("STOP")

    mean_weight_grad = torch.mul(new_returns, positive_weight_perturbations).mean(0)
    mean_bias_grad = torch.mul(new_returns, positive_bias_perturbations).mean(0)
    weights += learning_rate / noise_std_dev * mean_weight_grad
    biases += learning_rate / noise_std_dev * mean_bias_grad
