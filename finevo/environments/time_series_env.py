import numpy as np
import os
import pandas as pd
import torch
from ..base_object import BaseObject
from ..device_utils import set_device
from glob import glob
from gym import spaces
from tqdm import tqdm
from typing import Tuple


class TimeSeriesEnv(BaseObject):
    def __init__(
        self,
        instrument_name: str,
        num_envs: int = 2,
        num_intervals: int = 390,
        max_shares: int = 5,
        starting_balance: float = 10000,
        per_share_commission: float = 0.01,
        # NOTE: https://www.investopedia.com/ask/answers/05/shortmarginrequirements.asp
        # is a good resource for information on margin requirements.
        initial_margin_requirement: float = 1.5,
        maintenance_margin_requirement: float = 0.25,
        device_id: int = 0,
        testing_code: bool = False,
    ):
        self.instrument_name = instrument_name
        self.num_envs = num_envs
        self.num_intervals = num_intervals
        self.max_shares = max_shares
        self.starting_balance = starting_balance
        self.per_share_commission = per_share_commission
        self.initial_margin_requirement = initial_margin_requirement
        self.maintenance_margin_requirement = maintenance_margin_requirement
        self.log_return_scale_factor = 100
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
        self.training_dataframe: pd.DataFrame = pd.read_csv(
            path,
            names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"],
        )
        self.training_dataframe["Datetime"] = pd.to_datetime(
            self.training_dataframe["Date"] + " " + self.training_dataframe["Time"]
        )
        self.training_dataframe = self.training_dataframe.set_index("Datetime")

    def force_market_hours(self) -> None:
        self.training_dataframe = self.training_dataframe.between_time("9:30", "15:59")

    def set_up_environments(self) -> None:
        print(f"Setting up environments...")
        self.environments: "list[torch.Tensor]" = []
        self.prices_by_environment: "list[torch.Tensor]" = []
        unique_dates = self.training_dataframe["Date"].unique()
        num_dates = len(unique_dates)
        # use tqdm to track progress in setting up environments
        for _, date in zip(tqdm(range(num_dates)), unique_dates):
            (start_index, stop_index) = self.get_bounding_indices(date)
            if start_index < 0:
                continue
            blocked_df = self.training_dataframe[start_index:stop_index]
            single_env = blocked_df.drop(["Date", "Time", "Volume"], axis=1)
            (log_return_tensor, price_tensor) = self.get_tensors_from_dataframe(
                single_env
            )
            self.environments.append(log_return_tensor)
            self.prices_by_environment.append(price_tensor)

    def get_bounding_indices(self, date: str) -> Tuple[int, int]:
        date_indices = self.training_dataframe["Date"] == date
        start_index = self.training_dataframe.index.get_loc(
            self.training_dataframe.index[date_indices][0]
        )
        last_index = self.training_dataframe.index.get_loc(
            self.training_dataframe.index[date_indices][-1]
        )
        # need to backtrack by however many intervals we would like to see on each
        # step through the environment
        true_start_index = start_index - self.num_intervals
        return (true_start_index, last_index)

    def get_tensors_from_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        np_array = df.values
        self.values_per_interval = np_array.shape[1]
        price_tensor = torch.tensor(np_array, device=self.device)
        log_return_tensor = self.get_log_returns_from_price(price_tensor)
        return (log_return_tensor.flatten(), price_tensor.flatten())

    def get_log_returns_from_price(self, OHLC_tensor: torch.Tensor) -> torch.Tensor:
        log_return_tensor = torch.zeros(OHLC_tensor.shape, device=self.device)
        opens = OHLC_tensor[:, 0]
        for idx in range(1, 4):
            log_return_tensor[:, idx] = torch.log(OHLC_tensor[:, idx] / opens)
        previous_closes = OHLC_tensor[:-1, 3]
        # since the environment must begin at some price, just assume that it's the
        # same as the previous close price

        # FIXME: this is a bad way of doing things... we really should just make a
        # giant tensor from the training dataframe in the beginning, then make log
        # log returns out of that, then split it up appropriately into individual
        # environments

        previous_closes = torch.cat([opens[0].unsqueeze(0), previous_closes])
        log_return_tensor[:, 0] = torch.log(opens / previous_closes)
        return self.log_return_scale_factor * log_return_tensor

    def set_spaces(self) -> None:
        # observation dimension consists solely of price data
        self.num_obs = self.num_intervals * self.values_per_interval
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
        self.cash = self.starting_balance * torch.ones(
            (self.num_envs, 1), device=self.device
        )
        self.long_shares = torch.zeros((self.num_envs, 1), device=self.device)
        self.short_shares = torch.zeros((self.num_envs, 1), device=self.device)
        self.margin = torch.zeros((self.num_envs, 1), device=self.device)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        share_changes = self.get_share_changes_from_actions(actions)
        self.env_pointers += self.values_per_interval
        new_states = self.determine_new_states(share_changes)
        rewards = self.determine_immediate_rewards()
        dones = self.find_finished_environments()
        # if any environments have ended, all positions should be closed in preparation
        # for the next environment
        num_shares = self.short_shares + self.long_shares
        rewards -= dones * num_shares * self.per_share_commission
        self.reset_finished_environments(dones)
        return (new_states, rewards, dones, {})

    def get_share_changes_from_actions(self, actions: torch.Tensor) -> torch.Tensor:
        scaled_actions = actions * (self.max_shares + 0.5)
        share_changes = torch.round(scaled_actions)
        clamped_share_changes = share_changes.clamp(-self.max_shares, +self.max_shares)
        return clamped_share_changes

    def determine_new_states(self, share_changes: torch.Tensor) -> torch.Tensor:
        self.commissions = torch.zeros((self.num_envs, 1), device=self.device)
        self.set_current_prices()
        (
            positive_share_changes,
            negative_share_changes,
        ) = self.split_share_changes_by_sign(share_changes)
        # NOTE: the order here is important, because you want to ensure you've closed
        # all of your long position before entering a short position and vice-versa. In
        # other words, it shouldn't be possible to have both a long and a short
        # position at the same time.
        self.sell_long_positions(negative_share_changes)
        self.buy_back_short_positions(positive_share_changes)
        self.disallow_illegal_long_trades(positive_share_changes)
        self.initiate_long_trades(positive_share_changes)
        self.disallow_illegal_short_trades(negative_share_changes)
        self.initiate_short_trades(negative_share_changes)
        return self.reset()

    def set_current_prices(self) -> None:
        current_prices: "list[torch.Tensor]" = []
        for env_idx, env_ptr in zip(self.env_indices, self.env_pointers):
            prices = self.prices_by_environment[env_idx]
            current_price_data = torch.narrow(
                prices, 0, env_ptr.item(), self.values_per_interval
            )
            current_prices.append(current_price_data)
        self.current_prices = torch.stack(current_prices, dim=0)
        self.current_open_prices = torch.select(self.current_prices, 1, 0).view(
            self.num_envs, 1
        )
        self.current_high_prices = torch.select(self.current_prices, 1, 1).view(
            self.num_envs, 1
        )
        self.current_low_prices = torch.select(self.current_prices, 1, 2).view(
            self.num_envs, 1
        )
        self.current_close_prices = torch.select(self.current_prices, 1, 3).view(
            self.num_envs, 1
        )

    def split_share_changes_by_sign(
        self, share_changes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positive_share_changes = share_changes.clone()
        positive_share_changes[share_changes < 0] = 0
        negative_share_changes = share_changes.clone()
        negative_share_changes[share_changes > 0] = 0
        return (positive_share_changes, negative_share_changes)

    def sell_long_positions(self, negative_share_changes: torch.Tensor) -> None:
        new_long_shares = torch.relu(self.long_shares + negative_share_changes)
        sell_long_shares = self.long_shares - new_long_shares
        negative_share_changes += sell_long_shares
        self.increment_commissions(sell_long_shares)
        self.cash += sell_long_shares * (
            self.current_open_prices - self.per_share_commission
        )
        self.long_shares = new_long_shares

    def increment_commissions(self, num_shares: torch.Tensor) -> None:
        commissions = num_shares * self.per_share_commission
        self.commissions += commissions

    def buy_back_short_positions(self, positive_share_changes: torch.Tensor) -> None:
        new_short_shares = torch.relu(self.short_shares - positive_share_changes)
        buy_back_shares = self.short_shares - new_short_shares
        positive_share_changes -= buy_back_shares
        self.increment_commissions(buy_back_shares)
        self.cash -= buy_back_shares * (
            self.current_open_prices - self.per_share_commission
        )
        self.short_shares = new_short_shares
        new_margin = (
            self.initial_margin_requirement
            * self.short_shares
            * self.current_open_prices
        )
        change_in_margin = new_margin - self.margin
        self.cash -= change_in_margin
        self.margin = new_margin

    def disallow_illegal_long_trades(
        self, positive_share_changes: torch.Tensor
    ) -> None:
        new_long_positions = positive_share_changes * (
            self.current_open_prices + self.per_share_commission
        )
        new_cash = self.cash - new_long_positions
        positive_share_changes[new_cash < 0] = 0

    def initiate_long_trades(self, positive_share_changes: torch.Tensor) -> None:
        self.increment_commissions(positive_share_changes)
        self.cash -= positive_share_changes * (
            self.current_open_prices + self.per_share_commission
        )
        self.long_shares += positive_share_changes

    def disallow_illegal_short_trades(
        self, negative_share_changes: torch.Tensor
    ) -> None:
        short_commission = -negative_share_changes * self.per_share_commission
        new_short_positions = -negative_share_changes * self.current_open_prices
        initial_margin_requirement = (
            self.initial_margin_requirement * new_short_positions
        )
        new_cash = self.cash - initial_margin_requirement - short_commission
        negative_share_changes[new_cash < 0] = 0

    def initiate_short_trades(self, negative_share_changes: torch.Tensor) -> None:
        self.increment_commissions(-negative_share_changes)
        short_commission = -negative_share_changes * self.per_share_commission
        new_short_positions = -negative_share_changes * self.current_open_prices
        initial_margin_requirement = (
            self.initial_margin_requirement * new_short_positions
        )
        self.cash -= initial_margin_requirement + short_commission
        self.margin += initial_margin_requirement
        self.short_shares += -negative_share_changes

    def reset(self) -> torch.Tensor:
        return self.get_price_data_observations()

    def get_price_data_observations(self) -> torch.Tensor:
        data_observations: "list[torch.Tensor]" = []
        for env_idx, env_ptr in zip(self.env_indices, self.env_pointers):
            prices = self.prices_by_environment[env_idx]
            # NOTE: for optimization, it's actually possible to parallelize this with
            # the overloaded torch.narrow() function. However, it will require keeping
            # environments in a single tensor somehow...
            data_observation = torch.narrow(prices, 0, env_ptr.item(), self.num_obs)
            data_observations.append(data_observation)
        data_observation_tensors = torch.stack(data_observations, dim=0)
        return data_observation_tensors

    def determine_immediate_rewards(self) -> torch.Tensor:
        self.dones = self.cash < 0
        rewards = self.maintenance_margin_check(self.current_high_prices)
        self.margin_release(self.current_low_prices)
        rewards += self.maintenance_margin_check(self.current_close_prices)
        self.long_shares[self.dones] = 0
        self.short_shares[self.dones] = 0
        price_change = self.current_close_prices - self.current_open_prices
        rewards += (self.long_shares - self.short_shares) * price_change
        rewards -= self.commissions
        return rewards

    def maintenance_margin_check(self, current_prices: torch.Tensor) -> torch.Tensor:
        short_positions = self.short_shares * current_prices
        maintenance_margin_requirement = short_positions * (
            1 + self.maintenance_margin_requirement
        )
        margin_calls = torch.relu(maintenance_margin_requirement - self.margin)
        self.cash -= margin_calls
        self.margin += margin_calls
        self.dones = torch.logical_or(self.dones, self.cash < 0)
        return -margin_calls

    def margin_release(self, current_prices: torch.Tensor) -> None:
        short_positions = self.short_shares * current_prices
        initial_margin_requirement = short_positions * self.initial_margin_requirement
        margin_release = torch.relu(self.margin - initial_margin_requirement)
        self.margin -= margin_release
        self.cash += margin_release

    def find_finished_environments(self) -> torch.Tensor:
        for idx, (env_idx, env_ptr) in enumerate(
            zip(self.env_indices, self.env_pointers)
        ):
            env = self.environments[env_idx]
            if env_ptr.item() + self.num_obs >= env.shape[0]:
                self.dones[idx] = True
        return self.dones

    def reset_finished_environments(self, dones: torch.Tensor) -> None:
        self.cash[dones] = self.starting_balance
        self.margin[dones] = 0
        self.long_shares[dones] = 0
        self.short_shares[dones] = 0
        horizontal_dones = dones.view(4)
        self.env_indices[horizontal_dones] = np.random.randint(
            0, len(self.environments)
        )
        self.env_pointers[horizontal_dones] = 0

    def step_dataset(self, csv_key: str) -> None:
        self.print()
