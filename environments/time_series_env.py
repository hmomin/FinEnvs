import numpy as np
import pandas as pd
import torch
from glob import glob
from gym import spaces
from os.path import join
from pprint import pprint
from tqdm import tqdm
from typing import Tuple


class TimeSeriesEnv:
    def __init__(
        self,
        instrument_name: str,
        num_envs: int = 2,
        max_shares: int = 5,
        starting_balance: float = 1e4,
        num_intervals: int = 390,
        device_id: int = 0,
        testing_code: bool = False,
    ):
        self.instrument_name = instrument_name
        self.num_envs = num_envs
        self.max_shares = max_shares
        self.starting_balance = starting_balance
        self.num_intervals = num_intervals
        self.folder_name = self.get_folder_name(instrument_name)
        file_key = "dummy" if testing_code else "train"
        self.training_filename = self.find_file_by_key(file_key)
        self.set_device(device_id)
        self.process_training_data(self.training_filename)
        self.set_spaces()
        self.set_environment_params()

    def get_folder_name(self, folder_name: str) -> str:
        if "data" not in folder_name:
            folder_name = join("data", folder_name)
        return folder_name

    def find_file_by_key(self, key_string: str) -> str:
        training_filenames = glob(join(self.folder_name, "*" + key_string + "*"))
        if len(training_filenames) == 1:
            return training_filenames[0]
        else:
            raise Exception(
                f"More than one file was found in {self.folder_name} with key {key_string}"
            )

    def set_device(self, device_id: int):
        if torch.cuda.is_available():
            self.device = f"cuda:{device_id}"
        else:
            print(
                "WARNING: PyTorch is not recognizing a valid CUDA device -> forcing CPU..."
            )
            self.device = "cpu"

    def process_training_data(self, path: str):
        self.read_data(path)
        self.force_market_hours()
        self.set_up_environments()

    def read_data(self, path: str):
        print(f"Reading data for {self.instrument_name}...")
        self.df: pd.DataFrame = pd.read_csv(
            path,
            names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"],
        )
        self.df["Datetime"] = pd.to_datetime(self.df["Date"] + " " + self.df["Time"])
        self.df = self.df.set_index("Datetime")

    def force_market_hours(self):
        self.df = self.df.between_time("9:30", "15:59")

    def set_up_environments(self):
        print(f"Setting up environments for {self.instrument_name}...")
        self.environments: "list[torch.Tensor]" = []
        unique_dates = self.df["Date"].unique()
        num_dates = len(unique_dates)
        # use tqdm to track progress in setting up environments
        for _, date in zip(tqdm(range(num_dates)), unique_dates):
            (start_index, stop_index) = self.get_bounding_indices(date)
            if start_index < 0:
                continue
            single_env = self.df[start_index:stop_index].drop(["Date", "Time"], axis=1)
            single_env_tensor = self.get_tensor_from_dataframe(single_env)
            self.environments.append(single_env_tensor)

    def get_bounding_indices(self, date: str) -> Tuple[int, int]:
        date_indices = self.df["Date"] == date
        start_index = self.df.index.get_loc(self.df.index[date_indices][0])
        last_index = self.df.index.get_loc(self.df.index[date_indices][-1])
        # need to backtrack by however many intervals we would like to see on each
        # step through the environment
        true_start_index = start_index - self.num_intervals
        return (true_start_index, last_index)

    def get_tensor_from_dataframe(self, df: pd.DataFrame) -> torch.Tensor:
        np_array = df.values
        self.values_per_interval = np_array.shape[1]
        new_length = np_array.shape[0] * np_array.shape[1]
        flattened_np_array = np_array.reshape(new_length)
        return torch.tensor(flattened_np_array, device=self.device)

    def set_spaces(self):
        # observation dimension consists of:
        # {price data + balance + current number of shares}
        self.num_price_values = self.num_intervals * self.values_per_interval
        self.num_obs = self.num_price_values + 2
        # use a continuous action space and discretize based on max_shares
        self.num_acts = 1
        self.action_space = spaces.Box(
            np.ones(self.num_acts) * -1.0,
            np.ones(self.num_acts) * +1.0,
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            np.ones(self.num_obs) * -np.inf,
            np.ones(self.num_obs) * +np.inf,
            dtype=np.float64,
        )

    def set_environment_params(self):
        self.env_indices = torch.randint(
            0, len(self.environments), (self.num_envs,), device=self.device
        )
        self.env_pointers = torch.zeros(
            (self.num_envs,), dtype=torch.int64, device=self.device
        )
        self.balances = (
            torch.ones((self.num_envs, 1), device=self.device) * self.starting_balance
        )
        self.num_shares = torch.zeros((self.num_envs, 1), device=self.device)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        returns (
            observation tensor (N x S),
            reward tensor (N x 1),
            done tensor (N x 1),
            info dict (?)
        )
        """
        share_changes = self.get_share_changes_from_actions(actions)
        print(share_changes)
        # TODO: finish implementing this

    def get_share_changes_from_actions(self, actions: torch.Tensor) -> torch.Tensor:
        scaled_actions = actions * (self.max_shares + 0.5)
        share_changes = torch.round(scaled_actions)
        return share_changes.clamp(-self.max_shares, +self.max_shares)

    def reset(self) -> torch.Tensor:
        data_observations: "list[torch.Tensor]" = []
        for env_idx, env_ptr in zip(self.env_indices, self.env_pointers):
            env = self.environments[env_idx]
            # NOTE for optimization: it's actually possible to parallelize this with
            # the overloaded torch.narrow() function. It will require keeping
            # environments in a single tensor somehow...
            data_observation = torch.narrow(
                env, 0, env_ptr.item(), self.num_price_values
            )
            data_observations.append(data_observation)
        data_observation_tensors = torch.stack(data_observations, dim=0)
        observation_tensors = torch.cat(
            [data_observation_tensors, self.balances, self.num_shares], dim=1
        )
        return observation_tensors

    def reset_by_indices(self, indices: "list[int]") -> None:
        # NOTE for optimization: this could probably be parallelized somehow, but not a
        # high priority for now...
        for idx in indices:
            self.env_indices[idx] = np.random.randint(0, len(self.environments))
            self.env_pointers[idx] = 0
            self.balances[idx] = self.starting_balance
            self.num_shares[idx] = 0

    def print(self):
        pprint(vars(self))
