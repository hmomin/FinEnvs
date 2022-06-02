import csv
import numpy as np
from brax import envs
from brax.training import es
from time import time
from typing import Dict, Iterable

alphas: "list[float]" = np.logspace(-4, 0, 9, True, 10).tolist()
sigmas = list(alphas)
csv_name = "trials/brax_es_humanoid.csv"


def progress(_, metrics: Dict) -> None:
    minutes = (time() - start) / 60
    episode_return = metrics["eval/episode_reward"]
    write_to_csv([minutes, episode_return])
    # train_steps_per_sec.append(metrics["speed/sps"])


def write_to_csv(row_data: Iterable) -> None:
    with open(csv_name, "a") as f:
        writer = csv.writer(f)
        writer.writerow(row_data)


for alpha in alphas:
    for sigma in sigmas:
        write_to_csv(["NEW_TRIAL", alpha, sigma])
        # train_steps_per_sec = []
        start = time()

        es.train(
            environment_fn=envs.create_fn(env_name="humanoid"),
            num_timesteps=5_000_000,
            episode_length=1_000,
            action_repeat=1,
            l2coeff=0.005,
            max_devices_per_host=None,
            population_size=2_048,
            learning_rate=alpha,
            fitness_shaping=2,
            num_eval_envs=128,
            perturbation_std=sigma,
            seed=0,
            normalize_observations=False,
            log_frequency=1,
            center_fitness=False,
            progress_fn=progress,
        )

        # print(f"train steps/sec: {np.mean(train_steps_per_sec[1:])}")
