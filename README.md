# FinEvo: Comparing Scalable Neuroevolution with Deep Reinforcement Learning in Quantitative Finance Applications

For the current project proposal, see [this document](https://www.overleaf.com/read/pfjjvzcbpnmt).

## Long-Term Goals

- Create a massively parallel neuroevolution architecture that seamlessly integrates with symmetrically parallel simulation environments (time series, Isaac Gym, custom gym-like environments, etc.).
- Integrate the neuroevolution architecture to train on multiple GPUs.
- Establish state-of-the-art performance with evolutionary algorithms on parallelized quantitative finance tasks, Isaac Gym tasks, and related gym-like tasks (for example: Bipedal-Walker-Hardcore).
- Eventually, examine asynchronous updating between networks to prevent idle time among worker nodes.
- Eventually, examine a decentralized updating architecture to see if any benefits are provided by decentralization over centralization.

## Useful References

- [OpenAI Evolution Strategies paper](https://arxiv.org/pdf/1703.03864.pdf)
- [EvoJAX paper](https://arxiv.org/pdf/2202.05008.pdf)
- [Isaac Gym paper](https://arxiv.org/pdf/2108.10470.pdf)
- [PPO paper](https://arxiv.org/pdf/1707.06347.pdf)
- [DD-PPO paper](https://arxiv.org/pdf/1911.00357.pdf)
- [FinRL-Podracer paper](https://arxiv.org/pdf/2111.05188.pdf)
- [ElegantRL-Podracer paper](https://arxiv.org/pdf/2112.05923.pdf)
- [Deep Neuroevolution with GAs paper](https://arxiv.org/pdf/1712.06567.pdf)
- [Training Large Models on Many GPUs article](https://lilianweng.github.io/posts/2021-09-25-train-large/)
- [Evolutionary Models of Financial Markets survey](https://www.pnas.org/doi/pdf/10.1073/pnas.2104800118)
- [Offline Reinforcement Learning paper](https://arxiv.org/pdf/2005.01643.pdf)
