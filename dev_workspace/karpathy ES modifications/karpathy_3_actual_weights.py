"""
A bare bones examples of optimizing a black-box function using Natural Evolution
Strategies (NES), where the parameter distribution is a gaussian of fixed standard
deviation. -> parallelized with PyTorch and uses an actual weight matrix
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
print(f"solution: {str(solution)}")

for i in range(401):
    if i % 20 == 0:
        # fitness of current weights
        actions = torch.matmul(states, weights)
        print(f"iter: {i:4d} | R: {black_box_function(actions).item():0.6f}")

    perturbations = torch.randn((num_envs, 24, 3), device=device)
    duplicated_weights = weights.repeat(num_envs, 1, 1)
    perturbed_weights = duplicated_weights + noise_std_dev * perturbations
    new_actions = torch.matmul(duplicated_states, perturbed_weights)
    returns = black_box_function(new_actions).unsqueeze(-1)
    mean_grad = torch.mul(returns, perturbations).mean(0)
    weights += learning_rate / noise_std_dev * mean_grad
