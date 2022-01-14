from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import mlflow
import numpy as np


class Episode:

    def __init__(self):
        self.data = []

    def to_df(self):
        return pd.DataFrame(self.data)


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
            median_reward=df["reward"].median()
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

        mlflow.log_metrics(d)

    def add(self, episode: Episode):
        self._episodes.append(episode)

        if self._check_len():
            self.log()
            self._episodes = []


class TrackCallback(BaseCallback):

    def __init__(self, verbose=1):
        super(TrackCallback, self).__init__(verbose)

        self.data = []
        self._curr_episode = Episode()

        self.mlflow_logger = Logger()

    def to_df(self):
        return pd.concat([ep.to_df() for ep in self.data])

    def _on_step(self) -> bool:
        iteration = self.locals["iteration"]
        n_steps = self.locals["n_steps"]

        reward = self.locals["rewards"]
        action = self.locals["actions"]
        done = self.locals["dones"]
        infos = self.locals["infos"][0]

        infos["sb3_iteration"] = iteration
        infos["sb3_n_steps"] = n_steps
        infos["sb3_reward"] = reward
        infos["sb3_action"] = action
        infos["sb3_done"] = done

        self._curr_episode.data.append(infos)

        if done:
            self.data.append(self._curr_episode)
            self.mlflow_logger.add(self._curr_episode)
            self._curr_episode = Episode()

        return True
