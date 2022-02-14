import numpy as np

from rl.common.callbacks.logger import BaseLogCallback


class LogCallback(BaseLogCallback):

    def metrics(self, full_df, episode_end_df):
        return dict(
            mean_reward=full_df["reward"].mean(),
            median_reward=full_df["reward"].median(),
            sum_reward=full_df["reward"].sum(),
            auc_reward=np.trapz(full_df["reward"].to_numpy(), dx=1),
            mean_total_reward=full_df["total_reward"].mean(),
            median_total_reward=full_df["total_reward"].median(),
            ratio_execute=sum(full_df["sb3_action"] == 1) / len(full_df),
            inv_ratio=full_df["inv_ratio"].mean(),
            episode_end_trades_left_mean=episode_end_df["n_trades_left"].mean(),
            episode_end_trades_left_median=episode_end_df["n_trades_left"].median(),
        )
