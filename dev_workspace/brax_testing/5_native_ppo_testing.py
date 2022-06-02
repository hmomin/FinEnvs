import numpy as np
from brax import envs
from brax.training import ppo
from time import time

train_sps = []
start = time()


def progress(_, metrics):
    minutes = (time() - start) / 60
    episode_return = metrics["eval/episode_reward"]
    print(f"minutes: {minutes:.2f} | return: {episode_return:.2f}")
    train_sps.append(metrics["speed/sps"])


ppo.train(
    environment_fn=envs.create_fn(env_name="ant"),
    num_timesteps=150_000_000,
    log_frequency=10,
    reward_scaling=0.1,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=5,
    num_minibatches=32,
    num_update_epochs=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=2048,
    batch_size=1024,
    progress_fn=progress,
)

print(f"train steps/sec: {np.mean(train_sps[1:])}")
