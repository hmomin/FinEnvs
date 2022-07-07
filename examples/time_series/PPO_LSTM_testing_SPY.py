import os
import torch
from finenvs.agents.PPO.continuous_actor import ContinuousActorLSTM
from finenvs.environments.time_series_env import TimeSeriesEnv
from re import findall

relative_trials_dir = "../../trials/SPY_PPO_2022-07-06_20-22-21"


def get_trials_dir(relative_dir: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    trials_dir = os.path.join(current_dir, relative_dir)
    return trials_dir


def get_model_number(filename: str) -> int:
    numbers = findall("[0-9]+", filename)
    if len(numbers) != 1:
        raise Exception(f"Multiple numbers found in filename ({filename})")
    model_number = int(numbers[0])
    return model_number


def test_LSTM_on_evaluation_enviroments(env_name: str):
    env_keys = ["train", "valid", "test"]

    hidden_dim = 1024
    actor_shape = (5, hidden_dim, 1)
    test_actor = ContinuousActorLSTM(shape=actor_shape, sequence_length=4)

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
            env = TimeSeriesEnv(env_name, env_key, num_intervals=4, evaluate=True)
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
    test_LSTM_on_evaluation_enviroments("SPY")
