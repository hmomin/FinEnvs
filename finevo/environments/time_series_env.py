import numpy as np
import os
import pandas as pd
import torch

from ..base_object import BaseObject
from ..device_utils import set_device
from glob import glob
from gym import spaces
from pprint import pprint
from tqdm import tqdm
from typing import Tuple


class TimeSeriesEnv(BaseObject):
    def __init__(
        self,
        instrument_name: str,
        num_envs: int = 2,
        max_shares: int = 5,
        starting_balance: float = 10000,
        per_share_commission: float = 0.01,
        num_intervals: int = 390,
        device_id: int = 0,
        testing_code: bool = False,
    ):
        self.instrument_name = instrument_name
        self.num_envs = num_envs
        self.max_shares = max_shares
        self.starting_balance = starting_balance
        self.per_share_commission = per_share_commission
        self.num_intervals = num_intervals
        self.data_dir_name = self.get_data_dir_name(instrument_name)
        file_key = "dummy" if testing_code else "train"
        self.training_filename = self.find_file_by_key(file_key)
        self.device = set_device(device_id)
        self.process_training_data(self.training_filename)
        self.set_spaces()
        self.set_environment_params()

    def get_data_dir_name(self, data_dir_name: str) -> str:
        current_dir_name = os.path.dirname(os.path.realpath(__file__))
        if "data" not in data_dir_name:
            data_dir_name = os.path.join(current_dir_name, "..", "data", data_dir_name)
        return data_dir_name

    def find_file_by_key(self, key_string: str) -> str:
        training_filenames = glob(
            os.path.join(self.data_dir_name, "*" + key_string + "*")
        )
        num_files = len(training_filenames)
        if num_files == 0:
            raise Exception(
                f"No file was found in {self.data_dir_name} with key {key_string}"
            )
        elif num_files == 1:
            return training_filenames[0]
        else:
            raise Exception(
                f"More than one file was found in {self.data_dir_name} with key {key_string}"
            )

    def process_training_data(self, path: str) -> None:
        self.read_data(path)
        self.force_market_hours()
        self.set_up_environments()

    def read_data(self, path: str) -> None:
        print(f"Reading data for {self.instrument_name}...")
        self.df: pd.DataFrame = pd.read_csv(
            path,
            names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"],
        )
        self.df["Datetime"] = pd.to_datetime(self.df["Date"] + " " + self.df["Time"])
        self.df = self.df.set_index("Datetime")

    def force_market_hours(self) -> None:
        self.df = self.df.between_time("9:30", "15:59")

    def set_up_environments(self) -> None:
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

    def set_spaces(self) -> None:
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

    def set_environment_params(self) -> None:
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
        share_changes = self.get_share_changes_from_actions(actions)
        self.env_pointers += self.values_per_interval
        new_states = self.determine_new_states(share_changes)
        rewards = self.determine_immediate_rewards()
        dones = self.find_finished_environments()
        # If any environments have ended, all shares should be sold in preparation for
        # the next environment.
        rewards -= dones * self.num_shares * self.per_share_commission
        self.reset_finished_environments(dones)
        return (new_states, rewards, dones, {})

    def get_share_changes_from_actions(self, actions: torch.Tensor) -> torch.Tensor:
        scaled_actions = actions * (self.max_shares + 0.5)
        share_changes = torch.round(scaled_actions)
        clamped_share_changes = share_changes.clamp(-self.max_shares, +self.max_shares)
        return clamped_share_changes

    def determine_new_states(self, share_changes: torch.Tensor) -> torch.Tensor:
        self.set_current_prices()
        current_open_prices = torch.select(self.current_prices, 1, 0).view(
            self.num_envs, 1
        )
        self.ensure_legal_share_changes(share_changes, current_open_prices)
        self.true_share_changes = (
            torch.relu(self.num_shares + share_changes) - self.num_shares
        )
        capital_required = self.true_share_changes * current_open_prices
        self.balances -= capital_required
        self.num_shares += self.true_share_changes
        return self.reset()

    def set_current_prices(self) -> None:
        current_prices: "list[torch.Tensor]" = []
        for env_idx, env_ptr in zip(self.env_indices, self.env_pointers):
            env = self.environments[env_idx]
            current_price_data = torch.narrow(
                env, 0, env_ptr.item(), self.values_per_interval
            )
            current_prices.append(current_price_data)
        self.current_prices = torch.stack(current_prices, dim=0)

    def ensure_legal_share_changes(
        self, share_changes: torch.Tensor, current_open_prices: torch.Tensor
    ) -> None:
        not_enough_capital_mask = (
            torch.relu(share_changes) * current_open_prices > self.balances
        )
        share_changes[not_enough_capital_mask] = 0

    def reset(self) -> torch.Tensor:
        data_observation_tensors = self.get_price_data_observations()
        observation_tensors = torch.cat(
            [data_observation_tensors, self.balances, self.num_shares], dim=1
        )
        return observation_tensors

    def get_price_data_observations(self) -> torch.Tensor:
        data_observations: "list[torch.Tensor]" = []
        for env_idx, env_ptr in zip(self.env_indices, self.env_pointers):
            env = self.environments[env_idx]
            # NOTE: for optimization, it's actually possible to parallelize this with
            # the overloaded torch.narrow() function. It will require keeping
            # environments in a single tensor somehow...
            data_observation = torch.narrow(
                env, 0, env_ptr.item(), self.num_price_values
            )
            data_observations.append(data_observation)
        data_observation_tensors = torch.stack(data_observations, dim=0)
        return data_observation_tensors

    def determine_immediate_rewards(self) -> torch.Tensor:
        current_open_prices = torch.select(self.current_prices, 1, 0).view(
            self.num_envs, 1
        )
        current_close_prices = torch.select(self.current_prices, 1, 3).view(
            self.num_envs, 1
        )
        price_change = current_close_prices - current_open_prices
        balances_change = self.num_shares * price_change
        total_commision = torch.abs(self.true_share_changes) * self.per_share_commission
        return balances_change - total_commision

    def find_finished_environments(self) -> torch.Tensor:
        dones = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        for idx, (env_idx, env_ptr) in enumerate(
            zip(self.env_indices, self.env_pointers)
        ):
            env = self.environments[env_idx]
            if env_ptr.item() + self.num_price_values >= env.shape[0]:
                dones[idx] = True
        return dones

    def reset_finished_environments(self, dones: torch.Tensor) -> None:
        horizontal_dones = dones.view(4)
        self.env_indices[horizontal_dones] = np.random.randint(
            0, len(self.environments)
        )
        self.env_pointers[horizontal_dones] = 0
        self.balances[dones] = self.starting_balance
        self.num_shares[dones] = 0
