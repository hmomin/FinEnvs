"""
SOURCE: https://gist.github.com/rh314/76b1bbfdfb6da80fc15c512b973b3bf3
"""

# Python version of: https://gist.github.com/davidbarber/16708b9135f13c9599f754f4010a0284
# as per blog post: https://davidbarber.github.io/blog/2017/04/03/variational-optimisation/
# also see https://www.reddit.com/r/MachineLearning/comments/63dhfc/r_evolutionary_optimization_as_a_variational/

from __future__ import print_function

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import os, sys

png1 = "figures/variational-optimization1.png"
png2 = "figures/variational-optimization2.png"


def E(W, x, y):
    return (0.5 * (W.dot(x) - y) ** 2).mean()


def gradE(W, x, y):
    G = np.tile(W.dot(x) - y, (W.shape[1], 1)) * x
    g = G.sum(axis=1) / G.shape[1]
    g = g.T[None, :]
    return g


# Variational Optimisation
# f(x) is a simple quadratic objective function (linear regression sq loss)
# p(x|theta) is a Gaussian


# Create the dataset:
N = 10  # Number of datapoints
D = 2  # Dimension of the data
W0 = np.random.randn(1, D) / D**0.5  # true linear regression weight
x = np.random.randn(D, N)  # inputs
y = W0.dot(x)  # outputs


# plot the error surface:
NW = 50
w_low = -5
w_high = 5
w1 = np.linspace(w_low, w_high, NW)
w2 = w1
Esurf = np.zeros((NW, NW))
for i in range(NW):
    for j in range(NW):
        Esurf[i, j] = E(np.c_[w1[i], w2[j]], x, y)


Winit = np.array([-4, 4])[None, :]  # initial starting point for the optimisation

################################################################################

# standard gradient descent:
print("SGD")
Nloops = 150  # number of iterations
eta = 0.1  # learning rate
W = Winit + 0
Whist = []
for i in range(Nloops):
    print(E(W, x, y))
    Whist.append(W[0, :])
    # plot3(W(2),W(1),E(W,x,y)+0.1,'y.','markersize',20);
    gradE_curr = gradE(W, x, y)
    W = W - eta * gradE_curr
Whist = np.array(Whist)


def plot_history(Whist, aspect):
    plt.imshow(
        Esurf.T,
        interpolation="None",
        origin="lower",
        aspect=aspect,
        extent=[w_low, w_high, w_low, w_high],
    )
    plt.plot(Whist[:, 0], Whist[:, 1], ".r")
    plt.grid()
    plt.axis([w_low, w_high, w_low, w_high])
    extent = [x.min(), x.max(), y.min(), y.max()]


plot_history(Whist, 1)
plt.savefig(png1)

################################################################################
print("VAR OPT")
# Variational Optimisation:
Nsamples = 150  # number of samples
sd = np.array([[5]])  # initial standard deviation of the Gaussian
beta = 2 * np.log(sd)  # parameterise the standard variance
mu = Winit + 0  # initial mean of the Gaussian
sdvals = np.array(sd)
EvalVarOpt = np.zeros(Nloops)
f = np.zeros(Nloops)
mu_hist = [mu[0, :]]
for i in range(Nloops):
    # plot3(mu(2),mu(1),E(mu,x,y)+0.1,'r.','markersize',20);
    EvalVarOpt[i] = E(mu, x, y)  # error value
    print(E(mu, x, y))
    xsample = np.tile(mu, (Nsamples, 1)) + sd * np.random.randn(
        Nsamples, D
    )  # draw samples

    g = np.zeros((1, D))  # initialise the gradient for the mean mu
    gbeta = 0  # initialise the gradient for the standard deviation (beta par)
    for j in range(Nsamples):
        f[j] = E(xsample[[j], :], x, y)  # function value (error)
        g = g + (xsample[[j], :] - mu) * f[j] / sd**2
        gbeta = gbeta + 0.5 * f[[j]].dot(
            np.exp(-beta) * np.sum((xsample[[j], :] - mu) ** 2) - D
        )

    g = g / Nsamples
    gbeta = gbeta / Nsamples

    mu = mu - eta * g  # Stochastic gradient descent for the mean
    mu_hist.append(mu[0, :])
    beta = beta - 0.01 * gbeta  # Stochastic gradient descent for the variance par
    # comment the line above to turn off variance adaptation

    sd = np.exp(beta) ** 0.5
    sdvals = np.r_[sdvals, sd]

mu_hist = np.array(mu_hist)

plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 2)
plt.plot(sdvals)
plt.subplot(1, 2, 1)
plot_history(mu_hist, 2)
plt.savefig(png2, dpi=80)

print("Done. Check the PNG files:")
print("  %s\n  %s" % (png1, png2))
