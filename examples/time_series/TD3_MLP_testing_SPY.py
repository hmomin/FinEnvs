import os
import torch
from finenvs.agents.TD3.actor import ActorMLP
from finenvs.environments.time_series_env import TimeSeriesEnv
from re import findall

relative_trials_dir = "../trials/SPY_TD3_2022-06-20_23-03-21"


def get_trials_dir(relative_dir: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    trials_dir = os.path.join(current_dir, relative_dir)
    return trials_dir


def get_model_number(filename: str) -> int:
    numbers = findall("[0-9]+", filename)
    model_number = int(numbers[1])
    return model_number


def test_MLP_on_evaluation_enviroments(env_name: str):
    env_keys = ["train", "valid", "test"]

    hidden_dims = (1024, 512, 256, 128)
    actor_shape = (1561, *hidden_dims, 1)
    test_actor = ActorMLP(actor_shape)

    trials_dir = get_trials_dir(relative_trials_dir)
    model_filenames = os.listdir(trials_dir)
    list.sort(model_filenames, key=get_model_number)

    all_evaluation_metrics: "list[list[float]]" = []
    for model_filename in model_filenames:
        model_location = os.path.join(trials_dir, model_filename)
        test_actor.load_state_dict(torch.load(model_location))
        model_number = get_model_number(model_filename)

        evaluation_metrics: "list[float]" = [model_number]
        for env_key in env_keys:
            env = TimeSeriesEnv(env_name, env_key, evaluate=True)
            states = env.reset()
            while True:
                actions = test_actor.forward(states.float()).detach()
                (next_states, _, _, info_dict) = env.step(actions)
                if "returns" in info_dict:
                    break
                states = next_states
            day_returns: torch.Tensor = info_dict["returns"]
            evaluation_metrics.append(day_returns.mean().item())
        print(evaluation_metrics)

        all_evaluation_metrics.append(evaluation_metrics)


if __name__ == "__main__":
    test_MLP_on_evaluation_enviroments("SPY")
