from collections import deque

import numpy as np
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

    def __init__(self, n_episodes_before_mean_log=10):
        self._n_episodes_before_mean_log = n_episodes_before_mean_log
        self._episodes = []

    def _check_len(self):
        return len(self._episodes) == self._n_episodes_before_mean_log

    def log(self):
        dfs = [ep.to_df() for ep in self._episodes]
        df = pd.concat(dfs)

        d = dict(
            mean_reward=df["reward"].mean(),
            median_reward=df["reward"].median(),
            mean_reward_completed_steps=df["reward_completed_steps"].mean(),
            median_reward_completed_steps=df["reward_completed_steps"].median(),
            mean_reward_discount_n_trades_left=df["reward_discount_n_trades_left"].mean(),
            median_reward_discount_n_trades_left=df["reward_discount_n_trades_left"].median(),
            mean_total_reward=df["total_reward"].mean(),
            median_total_reward=df["total_reward"].median()
        )

        episode_end_completed_steps = []
        episode_end_forced = []
        episode_end_trades_left = []

        for ep in self._episodes:
            last_entry = ep.data[len(ep.data) - 1]
            episode_end_completed_steps.append(last_entry["completed_steps"])
            episode_end_trades_left.append(last_entry["n_trades_left"])
            episode_end_forced.append(last_entry["intermediate_episode_end"])

        d["episode_end_completed_steps_mean"] = np.mean(np.array(episode_end_completed_steps))
        d["episode_end_trades_left_mean"] = np.mean(np.array(episode_end_trades_left))
        d["episode_end_forced_mean"] = np.mean(np.array(episode_end_forced))

        d["episode_end_completed_steps_median"] = np.median(np.array(episode_end_completed_steps))
        d["episode_end_trades_left_median"] = np.median(np.array(episode_end_trades_left))
        d["episode_end_forced_median"] = np.median(np.array(episode_end_forced))

        wandb.log(d)

    def add(self, episode: Episode):
        self._episodes.append(episode)

        if self._check_len():
            self.log()
            self._episodes = []


class TrackCallback(BaseCallback):

    def __init__(self, verbose=1):
        super(TrackCallback, self).__init__(verbose)

        self._data = deque(maxlen=11)
        self._curr_episode = Episode()

        self.wandb_logger = Logger()

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
