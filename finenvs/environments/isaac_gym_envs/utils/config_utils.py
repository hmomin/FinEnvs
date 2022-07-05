import os
import yaml
from ..tasks import isaacgym_task_map
from typing import Dict


def load_task_config(env_name: str) -> Dict:
    if env_name not in isaacgym_task_map:
        handle_illegal_environment(env_name)
    current_dir_name = os.path.dirname(os.path.realpath(__file__))
    config_root = os.path.join(current_dir_name, "..", "configs")
    config_filename = os.path.join(config_root, env_name + ".yaml")
    with open(config_filename) as config_file:
        task_config = yaml.load(config_file, Loader=yaml.SafeLoader)
    return task_config


def handle_illegal_environment(illegal_name: str):
    legal_environment_names = ""
    for env_name in isaacgym_task_map:
        legal_environment_names += env_name + "\n"
    raise NameError(
        f"Incorrect environment name '{illegal_name}' specified for Isaac Gym training.\n"
        + "Choose from one of the following:\n"
        + legal_environment_names
    )


def get_isaac_gym_env_args(env_name: str) -> Dict:
    task_config = load_task_config(env_name)
    env_config = task_config["env"]
    return {
        "env_name": task_config["name"],
        "num_envs": env_config["numEnvs"],
        "num_observations": env_config["numObservations"],
        "num_actions": env_config["numActions"],
        "sequence_length": 1,
    }
