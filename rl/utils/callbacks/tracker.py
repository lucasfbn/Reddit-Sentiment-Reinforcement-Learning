from collections import deque
from typing import Callable

import pandas as pd
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class Episode:

    def __init__(self):
        self.data = []

    def to_df(self, include_last=True):
        if include_last:
            return pd.DataFrame(self.data)
        else:
            return pd.DataFrame(self.data[:-1])


class Logger:

    def __init__(self, episodes_log_interval, log_func):
        self._episodes_log_interval = episodes_log_interval
        self._log_func = log_func
        self._episodes = []

    def _check_len(self):
        return len(self._episodes) == self._episodes_log_interval

    def log(self):
        full_dfs = pd.concat([ep.to_df() for ep in self._episodes])
        episode_end_dfs = pd.concat([ep.to_df().tail(1) for ep in self._episodes])

        wandb.log(self._log_func(full_dfs, episode_end_dfs))

    def add(self, episode: Episode):
        self._episodes.append(episode)

        if self._check_len():
            self.log()
            self._episodes = []


def _default_log_func(x, y):
    return {}


class TrackCallback(BaseCallback):

    def __init__(self, verbose=1, episodes_log_interval=10, log_func: Callable = _default_log_func):
        super(TrackCallback, self).__init__(verbose)

        self._data = deque(maxlen=episodes_log_interval + 1)
        self._curr_episode = Episode()

        self.wandb_logger = Logger(episodes_log_interval, log_func)

    @property
    def data(self):
        return list(self._data)

    def to_df(self):
        return pd.concat([ep.to_df() for ep in self._data])

    def _on_step(self) -> bool:
        infos = self.locals["infos"][0]
        infos["sb3_iteration"] = self.locals["iteration"]
        infos["sb3_n_steps"] = self.locals["n_steps"]
        infos["sb3_reward"] = self.locals["rewards"]
        infos["sb3_action"] = self.locals["actions"]
        infos["sb3_done"] = self.locals["dones"]

        self._curr_episode.data.append(infos)

        if infos["sb3_done"]:
            self._data.append(self._curr_episode)
            self.wandb_logger.add(self._curr_episode)
            self._curr_episode = Episode()

        return True
