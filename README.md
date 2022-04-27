![FinEvo Logo](./logo.png)

# FinEvo: Scalable Neuroevolution for Financial Reinforcement Learning

For the current project proposal, see [this document](https://docs.google.com/document/d/11DenLB5lEzhJzPuUMtsYvy56dfAtmA7erlrbALWvqxU).

## Long-Term Goals

- Create a massively parallel neuroevolution architecture that seamlessly integrates with symmetrically parallel simulation environments (time series, Isaac Gym, custom gym-like environments, etc.).
- Integrate the neuroevolution architecture to train on multiple GPUs.
- Establish state-of-the-art performance with evolutionary algorithms on parallelized quantitative finance tasks and related gym-like tasks.
- Show that our implementation of evolution strategies dramatically reduces hardware cost, as compared to OpenAI's implementation. For instance, in the original OpenAI ES paper, they needed 80 machines each with 18 CPU cores on AWS EC2 ($122.40/hour total for the cheapest machines) to solve Humanoid in ~10 minutes. We can do better with far fewer resources and lower compute costs by taking advantage of our parallelized policy parameters model with Isaac Gym.
- Examine the adversarial interaction \textit{between} independently trained agents in completely simulated market environments.
- Consider a decentralized updating architecture to see if any benefits are provided by decentralization over centralization.

## Useful References

- [OpenAI Evolution Strategies paper](https://arxiv.org/pdf/1703.03864.pdf)
- [PEARL paper](https://arxiv.org/pdf/2201.09568.pdf)
- [Deep Neuroevolution with GAs paper](https://arxiv.org/pdf/1712.06567.pdf)
- [Isaac Gym paper](https://arxiv.org/pdf/2108.10470.pdf)
- [Training Large Models on Many GPUs article](https://lilianweng.github.io/posts/2021-09-25-train-large/)
- [PPO paper](https://arxiv.org/pdf/1707.06347.pdf)
- [FinRL-Podracer paper](https://arxiv.org/pdf/2111.05188.pdf)
- [ElegantRL-Podracer paper](https://arxiv.org/pdf/2112.05923.pdf)
- [Deep Reinforcement Learning Versus Evolution Strategies Survey](https://arxiv.org/pdf/2110.01411.pdf)
- [EvoJAX paper](https://arxiv.org/pdf/2202.05008.pdf)
- [DD-PPO paper](https://arxiv.org/pdf/1911.00357.pdf)
- [Evolutionary Models of Financial Markets survey](https://www.pnas.org/doi/pdf/10.1073/pnas.2104800118)
- [Offline Reinforcement Learning paper](https://arxiv.org/pdf/2005.01643.pdf)
