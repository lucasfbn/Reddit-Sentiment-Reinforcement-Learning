import pandas as pd
import wandb

from rl.utils.callbacks.base import Callback


class Episode:

    def __init__(self):
        self.data = []

    def to_df(self, include_last=True):
        if include_last:
            return pd.DataFrame(self.data)
        else:
            return pd.DataFrame(self.data[:-1])

    def __len__(self):
        return len(self.data)


class BaseLogCallback(Callback):

    def __init__(self, episodes_log_interval):
        super().__init__(episodes_log_interval)

        self.data = []
        self._curr_episode = Episode()

    def per_step(self):
        infos = self.locals["infos"][0]
        infos["sb3_iteration"] = self.locals["iteration"]
        infos["sb3_n_steps"] = self.locals["n_steps"]
        infos["sb3_reward"] = self.locals["rewards"]
        infos["sb3_action"] = self.locals["actions"]
        infos["sb3_done"] = self.locals["dones"]

        self._curr_episode.data.append(infos)

    def per_episode(self):
        self.data.append(self._curr_episode)
        self._curr_episode = Episode()

    def per_episode_interval(self):
        self.base_log()
        self.data = []

    def _base_metrics(self, full_df, episode_end_df):
        return dict(mean_sb3_reward=full_df["sb3_reward"].mean(),
                    mean_episode_len=sum(len(ep) for ep in self.data) / len(self.data))

    def base_log(self):
        full_df = pd.concat([ep.to_df() for ep in self.data])
        episode_end_df = pd.concat([ep.to_df().tail(1) for ep in self.data])

        base_metrics = self._base_metrics(full_df, episode_end_df)
        additional_metrics = self.metrics(full_df, episode_end_df)

        wandb.log(base_metrics | additional_metrics, step=self.num_timesteps)

    def metrics(self, full_df, episode_end_df):
        return {}
