"""
A bare bones examples of optimizing a black-box function using Natural Evolution
Strategies (NES), where the parameter distribution is a gaussian of fixed standard
deviation. -> parallelized with PyTorch and uses a full-on hidden layer
"""

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# hyperparameters
num_envs = 512
noise_std_dev = 1.0
learning_rate = 3e-4
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


# start the optimization with a random initial guess
torch.manual_seed(seed)
states = torch.randn((24,), device=device)
duplicated_states = states.repeat(num_envs, 1, 1)
weights = torch.randn((24, 3), device=device)
biases = torch.randn((1, 3), device=device)
activation = torch.nn.Identity()
print(f"solution: {str(solution)}")

for i in range(601):
    if i % 30 == 0:
        # fitness of current weights
        actions = activation(torch.matmul(states, weights) + biases)
        print(f"iter: {i:4d} | R: {black_box_function(actions).item():0.6f}")

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
