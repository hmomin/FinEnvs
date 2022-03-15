import os
import pandas as pd
import torch
from glob import glob
from pprint import pprint
from tqdm import tqdm
from typing import Tuple


class TimeSeriesEnv:
    def __init__(
        self,
        instrument_name: str,
        num_envs: int = 1,
        max_shares: int = 5,
        num_intervals: int = 390,
        device_id: int = 0,
    ):
        self.instrument_name = instrument_name
        self.num_envs = num_envs
        self.max_shares = max_shares
        self.num_intervals = num_intervals
        self.folder_name = self.get_folder_name(instrument_name)
        self.training_filename = self.find_file_by_key("train")
        self.set_device(device_id)
        self.process_training_data(self.training_filename)

    def get_folder_name(self, folder_name: str) -> str:
        if "data" not in folder_name:
            folder_name = os.path.join("data", folder_name)
        return folder_name

    def find_file_by_key(self, key_string: str) -> str:
        training_filenames = glob(
            os.path.join(self.folder_name, "*" + key_string + "*")
        )
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
        self.set_spaces()

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
        self.environments = []
        unique_dates = self.df["Date"].unique()
        num_dates = len(unique_dates)
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
        # use a continuous action space and discretize based on max_shares
        self.uses_continuous_action_space = True
        self.action_dim = 1
        self.observation_dim = self.num_intervals * self.values_per_interval

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
        # TODO: implement this

    def reset(self) -> torch.Tensor:
        """returns first observation tensor (N x S)"""
        # TODO: implement this

    def print(self):
        pprint(vars(self))
