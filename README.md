<p align="center">
    <img
        src="https://dr3ngl797z54v.cloudfront.net/FinEvol-logo.gif"
        alt="FinEvol Logo"
        width="75%"
    />
</p>

# FinEvol: Highly Parallel Neuroevolution with Applications in Quantitative Finance and Deep Reinforcement Learning Problems

For now, see [this Google Doc](https://docs.google.com/document/d/1ReqLKi2quQ19b96IR96vgH8gcuK-KlZLVckwZtJOBeM/edit#) for more information.

## Task Board

(maybe, all this would be better placed on a "Discussions" post)

HIGH PRIORITY:

- integrate Isaac Gym environment simulation with this repo (Assignee: Momin)
- create sample LSTM agent to train with PPO/TD3/SAC on time series environment and establish a baseline (Assignee: none)

MEDIUM PRIORITY:

- begin forming the architecture for evolutionary direct policy search on a single GPU/CPU (Assignee: none)
  - think about [EvoJAX](https://github.com/google/evojax)

LOW PRIORITY:

- begin planning to scale evolutionary architecture from a single GPU to multiple GPUs (Assignee: none)
  - think about tournament-based ensemble design in [ElegantRL-Podracer](https://arxiv.org/pdf/2112.05923.pdf) and decentralized distributed architecture in [DD-PPO](https://arxiv.org/pdf/1911.00357.pdf)

## Long-Term Goals

- create a massively parallel neuroevolution architecture that can seamlessly integrate with symmetrically parallel simulation environments (time series, Isaac Gym, custom gym-like environments, etc.)
- integrate and train with multiple GPUs
- establish state-of-the-art performance with evolutionary algorithms on Isaac Gym and related tasks (for example: Bipedal-Walker-Hardcore)
- eventually, examine a decentralized parallel architecture and see if any benefits are provided by decentralization over centralization

## Useful Papers

- [Isaac Gym paper](https://arxiv.org/pdf/2108.10470.pdf)
- [OpenAI Evolution Strategies paper](https://arxiv.org/pdf/1703.03864.pdf)
- [EvoJAX paper](https://arxiv.org/pdf/2202.05008.pdf)
- [PPO paper](https://arxiv.org/pdf/1707.06347.pdf)
- [DD-PPO paper](https://arxiv.org/pdf/1911.00357.pdf)
- [FinRL paper](https://arxiv.org/pdf/2011.09607.pdf)
- [ElegantRL-Podracer paper](https://arxiv.org/pdf/2112.05923.pdf)
- [Deep Neuroevolution with GAs paper](https://arxiv.org/pdf/1712.06567.pdf)
- [Evolutionary Models of Financial Markets survey](https://www.pnas.org/doi/pdf/10.1073/pnas.2104800118)
- [Offline Reinforcement Learning paper](https://arxiv.org/pdf/2005.01643.pdf)
