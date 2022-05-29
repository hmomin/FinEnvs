import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from os.path import join
from pandas import read_csv
from typing import Tuple

network_str = "MLP"
data_dir = f"Ant_{network_str}_trials"


def plot_data_for_seconds(filenames: "list[str]") -> None:
    seconds_all_trials = []
    evaluations_all_trials = []
    for filename in filenames:
        dataframe = read_csv(filename)
        seconds = zero_out(dataframe["unix_time"].to_list())
        evaluation_returns = dataframe["evaluation_return"].to_list()
        (interp_seconds, interp_eval_returns) = fill_in_missing(
            seconds, evaluation_returns, 1.0
        )
        seconds_all_trials.append(interp_seconds)
        evaluations_all_trials.append(interp_eval_returns)
    (seconds, avg_evaluation, low_evaluation, high_evaluation) = get_avg_and_bounds(
        seconds_all_trials, evaluations_all_trials
    )
    minutes = [t / 60 for t in seconds]
    plt.figure(figsize=(10, 7.5))
    plt.rc("font", weight="normal", size=20)
    plt.grid(
        visible=True,
        which="both",
        axis="both",
        color="k",
        linestyle="-",
        linewidth=0.1,
    )
    plt.plot(
        minutes, avg_evaluation, "k-",
    )
    plt.fill_between(minutes, low_evaluation, high_evaluation, color="turquoise")
    plt.xlim(xmin=0.0)
    plt.ylim(ymin=0.0)
    plt.title(f"PPO-{network_str} on Isaac Gym Ant Task ({len(filenames)} trials)")
    plt.xlabel("Elapsed Real Time (minutes)")
    plt.ylabel("Evaluation Return")
    plt.savefig(f"PPO_{network_str}_Isaac_Gym_Ant_by_time.png")
    plt.show()


def zero_out(times: "list[float]") -> "list[float]":
    starting_time = times[0]
    for idx, val in enumerate(times):
        times[idx] = val - starting_time
    return times


def fill_in_missing(
    x: "list[float]", y: "list[float]", increment: float
) -> Tuple["list[float]"]:
    interp_x = []
    interp_y = []
    x_val = np.ceil(x[0])
    idx = 0
    while x_val <= x[-1]:
        while x_val > x[idx + 1]:
            idx += 1
        y_val = np.interp(x_val, [x[idx], x[idx + 1]], [y[idx], y[idx + 1]])
        interp_x.append(x_val)
        interp_y.append(y_val)
        x_val += increment
    return (interp_x, interp_y)


def get_avg_and_bounds(
    seconds_all_trials: "list[list[float]]", evaluations_all_trials: "list[list[float]]"
) -> Tuple["list[float]"]:
    min_length = get_min_length_in_list(seconds_all_trials)
    seconds = []
    avg_evaluation = []
    low_evaluation = []
    high_evaluation = []
    for idx in range(min_length):
        seconds_val = seconds_all_trials[0][idx]
        (avg_eval_val, std_dev_eval_val) = get_avg_and_std_dev_by_index(
            evaluations_all_trials, idx
        )
        seconds.append(seconds_val)
        avg_evaluation.append(avg_eval_val)
        low_evaluation.append(avg_eval_val - std_dev_eval_val)
        high_evaluation.append(avg_eval_val + std_dev_eval_val)
    return (seconds, avg_evaluation, low_evaluation, high_evaluation)


def get_min_length_in_list(list_of_lists: "list[list[float]]") -> int:
    min_length = np.inf
    for some_list in list_of_lists:
        min_length = min(min_length, len(some_list))
    return min_length


def get_avg_and_std_dev_by_index(
    list_of_lists: "list[list[float]]", idx: int
) -> Tuple[float, float]:
    vals = []
    for some_list in list_of_lists:
        vals.append(some_list[idx])
    return (np.mean(vals), np.std(vals))


def plot_data_for_samples(filenames: "list[str]") -> None:
    samples_all_trials = []
    evaluations_all_trials = []
    for filename in filenames:
        dataframe = read_csv(filename)
        num_samples = dataframe["num_training_samples"].to_list()
        evaluation_returns = dataframe["evaluation_return"].to_list()
        (interp_samples, interp_eval_returns) = fill_in_missing(
            num_samples, evaluation_returns, 65536
        )
        samples_all_trials.append(interp_samples)
        evaluations_all_trials.append(interp_eval_returns)
    (samples, avg_evaluation, low_evaluation, high_evaluation) = get_avg_and_bounds(
        samples_all_trials, evaluations_all_trials
    )
    plt.figure(figsize=(10, 7.5))
    plt.rc("font", weight="normal", size=20)
    plt.grid(
        visible=True,
        which="both",
        axis="both",
        color="k",
        linestyle="-",
        linewidth=0.1,
    )
    plt.plot(
        samples, avg_evaluation, "k-",
    )
    plt.fill_between(samples, low_evaluation, high_evaluation, color="turquoise")
    plt.xlim(xmin=0.0)
    plt.ylim(ymin=0.0)
    plt.title(f"PPO-{network_str} on Isaac Gym Ant Task ({len(filenames)} trials)")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Evaluation Return")
    plt.savefig(f"PPO_{network_str}_Isaac_Gym_Ant_by_samples.png")
    plt.show()


if __name__ == "__main__":
    filenames = glob(join(data_dir, "*.csv"))
    list.sort(filenames)
    plot_data_for_seconds(filenames)
    plot_data_for_samples(filenames)
