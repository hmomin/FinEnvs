import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Sinc function
f = lambda x: -np.sinc(x)

batch_size = 10000
N = 50
mu = np.linspace(-5, 5, N)
sigma = np.linspace(1e-1, 2, N)

rand_no = np.random.randn(1, batch_size)[0]
# Create array with different mu and sigma
rand_array = np.add.outer(np.multiply.outer(rand_no, sigma), mu)
# E = summation of all random values / total no. of samples
expected_value = np.sum(f(rand_array), 0) / batch_size

# Plotting related code
MU, SIGMA = np.meshgrid(mu, sigma)

fig = plt.figure()
ax = fig.add_subplot(221, projection="3d")
surf = ax.plot_surface(
    SIGMA, MU, expected_value, cmap=cm.coolwarm, linewidth=0, antialiased=False
)

ax = fig.add_subplot(222)
ax.plot(expected_value[0, :])

ax = fig.add_subplot(223)
ax.contourf(MU, SIGMA, expected_value, cmap=cm.coolwarm)

plt.show()
