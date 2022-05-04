"""
A bare bones examples of optimizing a black-box function using Natural Evolution
Strategies (NES), where the parameter distribution is a gaussian of fixed standard
deviation. -> parallelized with PyTorch
"""

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# hyperparameters
num_envs = 512
noise_std_dev = 1.0
learning_rate = 3e-4
solution = torch.tensor((0.5, 0.1, -0.3), device=device)
seed = 0

# Here, we would normally:
# 1) create a neural network with weights,
# 2) run the neural network on the environment for some time,
# 3) and sum up and return the total reward...

# But, for the purposes of an example, lets try to minimize the L2 distance to a
# specific solution vector. So the highest reward we can achieve is 0, when our vector
# of weights is exactly equal to the solution.

expanded_solution = solution.unsqueeze(0)
# the black-box function we want to optimize
def black_box_function(weights: torch.Tensor) -> torch.Tensor:
    if len(weights.shape) == 1:
        weights.unsqueeze(0)
    reward = -torch.sum(torch.square(weights - expanded_solution), dim=1)
    return reward


# start the optimization with a random initial guess
torch.manual_seed(seed)
weights = torch.randn((3,), device=device)
print(f"solution: {str(solution)}")

for i in range(6001):
    if i % 300 == 0:
        # fitness of current weights
        print(
            f"iter: {i:4d} | W: {weights} | R: {black_box_function(weights).item():0.6f}"
        )

    perturbations = torch.randn((num_envs, 3), device=device)
    duplicated_weights = weights.repeat(num_envs, 1)
    perturbed_weights = duplicated_weights + noise_std_dev * perturbations
    returns = black_box_function(perturbed_weights).unsqueeze(1)
    mean_grad = torch.mul(returns, perturbations).mean(0)
    weights += learning_rate / noise_std_dev * mean_grad
