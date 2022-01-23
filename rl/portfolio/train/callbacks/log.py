import numpy as np

from rl.utils.callbacks.logger import BaseLogCallback


class LogCallback(BaseLogCallback):

    def metrics(self, full_df, episode_end_df):
        return dict(
            mean_reward=full_df["reward"].mean(),
            median_reward=full_df["reward"].median(),
            sum_reward=full_df["reward"].sum(),
            auc_reward=np.trapz(full_df["reward"].to_numpy(), dx=1),
            mean_reward_completed_steps=full_df["reward_completed_steps"].mean(),
            median_reward_completed_steps=full_df["reward_completed_steps"].median(),
            mean_reward_discount_n_trades_left=full_df["reward_discount_n_trades_left"].mean(),
            median_reward_discount_n_trades_left=full_df["reward_discount_n_trades_left"].median(),
            mean_total_reward=full_df["total_reward"].mean(),
            median_total_reward=full_df["total_reward"].median(),
            ratio_execute=sum(full_df["sb3_action"] == 1) / len(full_df),
            episode_end_completed_steps_mean=episode_end_df["completed_steps"].mean(),
            episode_end_trades_left_mean=episode_end_df["n_trades_left"].mean(),
            episode_end_forced_mean=episode_end_df["intermediate_episode_end"].mean(),
            episode_end_completed_steps_median=episode_end_df["completed_steps"].median(),
            episode_end_trades_left_median=episode_end_df["n_trades_left"].median(),
            episode_end_forced_median=episode_end_df["intermediate_episode_end"].median(),
        )
