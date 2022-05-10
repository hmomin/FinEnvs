"""
Using Evolution Strategies to solve the CartPole environment
1. start with random policy
2. generate n new policies by perturbing weights of original
    policy with gaussian noise
3. evaluate the n new policies (run episode and get reward for each)
4. update weights of original policy according to gradient of reward
"""

import numpy as np
import gym

# neural network to store state to action policy
IL = 4  # input layer nodes
HL = 20  # hidden layer nodes
OL = 2  # output layer nodes
w1 = np.random.randn(HL, IL) / np.sqrt(IL)
w2 = np.random.randn(OL, HL) / np.sqrt(HL)

# feed-forward propagation
def predict(s, w1, w2):
    h = np.dot(w1, s)  # input to hidden layer
    h[h < 0] = 0  # relu
    out = np.dot(w2, h)  # hidden layer to output
    out = 1.0 / (1.0 + np.exp(-out))  # sigmoid
    return out


NumEpisodes = 100

# load environment
env = gym.make("CartPole-v0")

# hyperparameters
NumPolicies = 512
NumWeights1 = len(w1.flatten())
NumWeights2 = len(w2.flatten())

sigma = 0.1
learning_rate = 3e-4

Reward = np.zeros(NumPolicies)

# start learning
for episode in range(NumEpisodes):

    # generate random variations around original policy
    eps = np.random.randn(
        NumPolicies, NumWeights1 + NumWeights2
    )  # samples from a normal distribution N(0,1)

    # evaluate each policy over one episode
    for policy in range(NumPolicies):

        w1_try = w1 + sigma * eps[policy, :NumWeights1].reshape(w1.shape)
        w2_try = w2 + sigma * eps[policy, NumWeights1:].reshape(w2.shape)

        # initial state
        observation = env.reset()  # observe initial state

        Reward[policy] = 0

        while True:

            # normalize inputs
            # observation[0] /= 2.5
            # observation[1] /= 2.5
            # observation[2] /= 0.2
            # observation[3] /= 2.5

            Action = predict(observation, w1_try, w2_try)
            Action = np.argmax(Action)

            # execute action
            observation_new, reward, done, _ = env.step(Action)

            Reward[policy] += reward

            # update state
            observation = observation_new

            # end episode
            if done:
                break

    # calculate incremental rewards
    F = Reward - np.mean(Reward)

    # update weights of original policy according to rewards of all variations
    weights_update = learning_rate / (NumPolicies * sigma) * np.dot(eps.T, F)

    w1 += weights_update[:NumWeights1].reshape(w1.shape)
    w2 += weights_update[NumWeights1:].reshape(w2.shape)

    # if episode%100 == 0:
    print("Episode {0}, reward = {1}".format(episode, np.mean(Reward)))

env.close()
