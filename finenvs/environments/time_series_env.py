import json
import numpy as np
import os
import pandas as pd
import torch
from ..agents.PPO.continuous_actor import ContinuousActorMLP
from ..base_object import BaseObject
from ..device_utils import set_device
from glob import glob
from gym import spaces
from tqdm import tqdm
from typing import Dict, Tuple


class TimeSeriesEnv(BaseObject):
    def __init__(
        self,
        instrument_name: str,
        dataset_key: str = "dummy",
        num_envs: int = 1,
        num_intervals: int = 390,
        max_shares: int = 5,
        starting_balance: float = 10000,
        per_share_commission: float = 0.01,
        # NOTE: https://www.investopedia.com/ask/answers/05/shortmarginrequirements.asp
        # is a good resource for information on margin requirements.
        initial_margin_requirement: float = 1.5,
        maintenance_margin_requirement: float = 0.25,
        evaluate: bool = False,
        device_id: int = 0,
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
        self.file_key = self.determine_file_key(dataset_key)
        self.filename = self.find_file_by_key()
        self.evaluate = evaluate
        self.device = set_device(device_id)
        self.process_data(self.filename)
        self.set_spaces()
        self.set_environment_params()

    def get_data_dir_name(self, data_dir_name: str) -> str:
        if "data" not in data_dir_name:
            current_dir_name = os.path.dirname(os.path.realpath(__file__))
            data_dir_name = os.path.join(current_dir_name, "..", "data", data_dir_name)
        return data_dir_name

    def determine_file_key(self, key_attempt: str) -> str:
        possible_keys = ["dummy", "train", "valid", "test"]
        for possible_key in possible_keys:
            if possible_key in key_attempt:
                return possible_key
        raise Exception(f"dataset_key expected to be one of: " + str(possible_keys))

    def find_file_by_key(self) -> str:
        key_string = self.file_key
        filenames = glob(os.path.join(self.data_dir_name, "*" + key_string + "*.csv"))
        num_files = len(filenames)
        if num_files == 0:
            raise Exception(
                f"No file was found in {self.data_dir_name} with key ({key_string})"
            )
        elif num_files == 1:
            return filenames[0]
        else:
            raise Exception(
                f"More than one file was found in {self.data_dir_name} with key ({key_string})"
            )

    def process_data(self, path: str) -> None:
        self.read_data(path)
        self.force_market_hours()
        self.set_up_environments()

    def read_data(self, path: str) -> None:
        self.dataframe: pd.DataFrame = pd.read_csv(
            path,
            names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"],
        )
        self.dataframe["Datetime"] = pd.to_datetime(
            self.dataframe["Date"] + " " + self.dataframe["Time"]
        )
        self.dataframe = self.dataframe.set_index("Datetime")

    def force_market_hours(self) -> None:
        self.dataframe = self.dataframe.between_time("9:30", "15:59")

    def set_up_environments(self) -> None:
        self.log_return_environments: torch.Tensor = None
        self.price_environments: torch.Tensor = None
        unique_dates = self.dataframe["Date"].unique()
        (start_indices, stop_indices, max_length) = self.get_environment_bounds(
            unique_dates
        )
        self.generate_torch_datasets()
        self.generate_environments(start_indices, stop_indices, max_length)

    def get_environment_bounds(self, dates: np.ndarray) -> Tuple[list, list, int]:
        bound_indices_name = os.path.join(
            self.data_dir_name, f"{self.file_key}_bounds_cache.json"
        )
        try:
            with open(bound_indices_name) as f:
                cache = json.load(f)
            return (
                cache["start_indices"],
                cache["stop_indices"],
                cache["max_length"],
            )
        except:
            print(f"Determining environment bounds...")
            (
                start_indices,
                stop_indices,
                max_length,
            ) = self.determine_environment_bounds(dates)
            self.cache_indices(
                bound_indices_name, start_indices, stop_indices, max_length
            )
            return (start_indices, stop_indices, max_length)

    def determine_environment_bounds(self, dates: np.ndarray) -> Tuple[list, list, int]:
        start_indices = []
        stop_indices = []
        max_length = 0
        date_counter = range(len(dates))
        for _, date in zip(tqdm(date_counter), dates):
            (start_index, stop_index) = self.get_bounding_indices(date)
            if start_index < 0:
                continue
            max_length = max(max_length, stop_index - start_index + 1)
            start_indices.append(start_index)
            stop_indices.append(stop_index)
        return (start_indices, stop_indices, max_length)

    def get_bounding_indices(self, date: str) -> Tuple[int, int]:
        date_indices = self.dataframe["Date"] == date
        start_index = self.dataframe.index.get_loc(
            self.dataframe.index[date_indices][0]
        )
        last_index = self.dataframe.index.get_loc(
            self.dataframe.index[date_indices][-1]
        )
        # need to backtrack by however many intervals we would like to see on each
        # step through the environment
        true_start_index = start_index - self.num_intervals
        return (true_start_index, last_index)

    def cache_indices(
        self, filename: str, starts: "list[int]", stops: "list[int]", max_length: int
    ) -> None:
        cache = {
            "start_indices": starts,
            "stop_indices": stops,
            "max_length": max_length,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)

    def generate_torch_datasets(self) -> None:
        self.generate_price_dataset()
        self.generate_log_return_dataset()

    def generate_price_dataset(self) -> None:
        reduced_dataframe = self.dataframe.drop(["Date", "Time", "Volume"], axis=1)
        numpy_dataset = reduced_dataframe.values
        self.values_per_interval = numpy_dataset.shape[1]
        self.dataset = torch.tensor(
            numpy_dataset,
            dtype=torch.float64,
            device=self.device,
        )

    def generate_log_return_dataset(self) -> None:
        log_return_dataset = torch.zeros(
            self.dataset.shape, dtype=torch.float64, device=self.device
        )
        opens = self.dataset[:, 0]
        # high, low, and close returns can be calculated using only the open values
        for idx in range(1, 4):
            log_return_dataset[:, idx] = torch.log(self.dataset[:, idx] / opens)
        # open returns require using the previous close values
        previous_closes = self.dataset[:-1, 3]
        # assume the previous close for the first open value is the same
        previous_closes = torch.cat([opens[0].unsqueeze(0), previous_closes])
        log_return_dataset[:, 0] = torch.log(opens / previous_closes)
        self.log_return_dataset: torch.Tensor = (
            self.log_return_scale_factor * log_return_dataset
        )

    def generate_environments(
        self, start_indices: "list[int]", stop_indices: "list[int]", max_length: int
    ) -> None:
        price_environment_list: "list[torch.Tensor]" = []
        log_return_environment_list: "list[torch.Tensor]" = []
        for start_index, stop_index in zip(start_indices, stop_indices):
            price_env = self.dataset[start_index : stop_index + 1, :]
            log_return_env = self.log_return_dataset[start_index : stop_index + 1, :]
            env_length = price_env.shape[0]
            remaining_length = max_length - env_length
            if remaining_length > 0:
                nans = torch.rand(
                    (remaining_length, price_env.shape[1]), device=self.device
                ).float()
                nans[:] = float("nan")
                price_env = torch.cat([price_env, nans], dim=0)
                log_return_env = torch.cat([log_return_env, nans], dim=0)
            price_environment_list.append(price_env)
            log_return_environment_list.append(log_return_env)
        self.price_environments = torch.stack(price_environment_list, dim=0)
        self.log_return_environments = torch.stack(log_return_environment_list, dim=0)

    def set_spaces(self) -> None:
        self.num_price_values = self.num_intervals * self.values_per_interval
        # observation dimension consists of:
        # {price data + current position}
        self.num_obs = self.num_price_values + 1
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

    def get_env_args(self) -> Dict:
        return {
            "env_name": self.instrument_name,
            "num_envs": self.num_envs,
            "num_observations": self.num_obs,
            "num_actions": self.num_acts,
        }

    def set_environment_params(self) -> None:
        if self.evaluate:
            self.env_indices = torch.arange(
                0, self.price_environments.shape[0], 1, device=self.device
            )
            self.num_envs = self.price_environments.shape[0]
            self.reset_evaluation_metrics()
        else:
            self.env_indices = torch.randint(
                0,
                self.price_environments.shape[0],
                (self.num_envs,),
                device=self.device,
            )
            self.env_indices[-1] = 0
        self.env_pointers = torch.zeros(
            (self.num_envs,), dtype=torch.int64, device=self.device
        )
        self.env_spots = torch.arange(0, self.num_intervals, device=self.device).repeat(
            self.num_envs, 1
        )
        self.cash = self.starting_balance * torch.ones(
            (self.num_envs, 1), device=self.device
        )
        self.long_shares = torch.zeros((self.num_envs, 1), device=self.device)
        self.short_shares = torch.zeros((self.num_envs, 1), device=self.device)
        self.margin = torch.zeros((self.num_envs, 1), device=self.device)

    def reset_evaluation_metrics(self) -> None:
        self.terminated_episodes = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self.episode_returns = torch.zeros((self.num_envs,), device=self.device)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        share_changes = self.get_share_changes_from_actions(actions)
        self.env_pointers += 1
        self.env_spots += 1
        new_states = self.determine_new_states(share_changes)
        rewards = self.determine_immediate_rewards()
        dones = self.find_finished_environments()
        # if any environments have ended, all positions should be closed in preparation
        # for the next environment
        num_shares = self.short_shares + self.long_shares
        rewards -= dones * num_shares * self.per_share_commission
        self.reset_finished_environments(dones)
        rewards = rewards.squeeze(1)
        dones = dones.squeeze(1)
        info_dict = (
            self.record_evaluation_metrics(rewards, dones) if self.evaluate else {}
        )
        return (new_states, rewards, dones.int(), info_dict)

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
        current_price_environments = self.price_environments[self.env_indices, :, :]
        # some torch.gather() magic below to find the current prices...
        last_price_spots = self.env_spots[:, -1].unsqueeze(1)
        current_index_spots = last_price_spots.repeat(1, 4).unsqueeze(1)
        current_prices = torch.gather(
            current_price_environments, dim=1, index=current_index_spots
        ).squeeze(1)
        self.current_open_prices = torch.select(current_prices, 1, 0).view(
            self.num_envs, 1
        )
        self.current_high_prices = torch.select(current_prices, 1, 1).view(
            self.num_envs, 1
        )
        self.current_low_prices = torch.select(current_prices, 1, 2).view(
            self.num_envs, 1
        )
        self.current_close_prices = torch.select(current_prices, 1, 3).view(
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
            self.current_open_prices + self.per_share_commission
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
        raw_log_return_data = self.get_log_return_observations()
        # flatten for MLP architectures
        (x, y, z) = raw_log_return_data.shape
        log_return_data = torch.reshape(raw_log_return_data, (x, y * z))
        if not hasattr(self, "current_close_prices"):
            self.set_current_prices()
        current_positions = (
            self.long_shares - self.short_shares
        ) * self.current_close_prices
        scaled_positions = current_positions / self.starting_balance
        observation_tensors = torch.cat([log_return_data, scaled_positions], dim=1)
        return observation_tensors

    def get_log_return_observations(self) -> torch.Tensor:
        current_log_return_environments = self.log_return_environments[
            self.env_indices, :, :
        ]
        current_index_spots = self.env_spots.unsqueeze(-1).repeat(1, 1, 4)
        log_return_observations = torch.gather(
            current_log_return_environments, dim=1, index=current_index_spots
        )
        return log_return_observations

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
        current_environments = self.log_return_environments[self.env_indices, :, :]
        max_length = current_environments.shape[1]
        next_spots = (self.env_spots[:, -1] + 1).unsqueeze(-1)
        self.dones = torch.logical_or(self.dones, next_spots >= max_length)
        next_spots = next_spots.squeeze(1)
        self.find_nan_spots(next_spots)
        return self.dones

    def find_nan_spots(self, next_spots: torch.Tensor) -> None:
        not_dones = ~self.dones.squeeze(1)
        not_done_indices = not_dones.nonzero()
        env_indices_to_check = self.env_indices[not_dones]
        spots_to_check = next_spots[not_dones]
        unchecked_envs = self.log_return_environments[env_indices_to_check, :, :]
        spots_to_check = spots_to_check.unsqueeze(-1).unsqueeze(-1)
        next_open_values = torch.gather(unchecked_envs, dim=1, index=spots_to_check)
        nan_spots = torch.isnan(next_open_values).squeeze(2).squeeze(1)
        nans_found = not_done_indices[nan_spots]
        self.dones[nans_found] = True

    def reset_finished_environments(self, dones: torch.Tensor) -> None:
        self.cash[dones] = self.starting_balance
        self.margin[dones] = 0
        self.long_shares[dones] = 0
        self.short_shares[dones] = 0
        horizontal_dones = dones.view(self.num_envs)
        if not self.evaluate:
            last_env_index = self.env_indices[-1].item()
            num_envs = self.price_environments.shape[0]
            self.env_indices[horizontal_dones] = torch.randint(
                0,
                num_envs,
                self.env_indices[horizontal_dones].shape,
                device=self.device,
            )
            # the last environment should be reserved for evaluation purposes: it'll
            # cycle through each of the days in sequence, instead of choosing randomly
            if horizontal_dones[-1].item():
                self.env_indices[-1] = (last_env_index + 1) % num_envs
        self.env_pointers[horizontal_dones] = 0
        starting_spots = self.env_spots[:, 0]
        env_spot_deltas = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        env_spot_deltas[horizontal_dones] = starting_spots[horizontal_dones]
        env_spot_deltas = env_spot_deltas.unsqueeze(-1).repeat(1, self.num_intervals)
        self.env_spots -= env_spot_deltas

    def record_evaluation_metrics(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> Dict:
        # ignore rewards from terminated episodes
        terminated_indices = self.terminated_episodes.nonzero()
        rewards[terminated_indices] = 0
        self.terminated_episodes[dones] = True
        self.episode_returns += rewards
        if torch.all(self.terminated_episodes):
            info_dict = {"returns": self.episode_returns}
            self.reset_evaluation_metrics()
            return info_dict
        else:
            return {}
