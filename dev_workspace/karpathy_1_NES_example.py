"""
A bare bones examples of optimizing a black-box function using Natural Evolution
Strategies (NES), where the parameter distribution is a gaussian of fixed standard
deviation.
"""

import numpy as np

# hyperparameters
population_size = 512
noise_std_dev = 1.0
learning_rate = 3e-4
solution = np.array([0.5, 0.1, 0.3])
seed = 0

# Here, we would normally:
# 1) create a neural network with weights,
# 2) run the neural network on the environment for some time,
# 3) and sum up and return the total reward...

# But, for the purposes of an example, lets try to minimize the L2 distance to a
# specific solution vector. So the highest reward we can achieve is 0, when our vector
# of weights is exactly equal to the solution.

# the black-box function we want to optimize
def black_box_function(weights: np.ndarray) -> float:
    reward = -np.sum(np.square(solution - weights))
    return reward


print(f"solution: {str(solution)}")
# start the optimization with a random initial guess
np.random.seed(seed)
weights = np.random.randn(3)

for i in range(6001):
    if i % 300 == 0:
        # print current fitness of the most likely parameter setting
        print(
            f"iter: {i:4d} | W: {str(weights)} | R: {black_box_function(weights):0.6f}"
        )

    perturbations = np.random.randn(population_size, 3)
    returns = np.zeros(population_size)
    for j in range(population_size):
        perturbed_weights = weights + noise_std_dev * perturbations[j]
        returns[j] = black_box_function(perturbed_weights)

    weights += (
        learning_rate
        / (population_size * noise_std_dev)
        * np.dot(perturbations.T, returns)
    )
