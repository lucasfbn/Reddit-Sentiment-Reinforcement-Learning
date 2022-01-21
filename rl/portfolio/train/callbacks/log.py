def log_func(full_df, episode_end_df):
    return dict(
        mean_reward=full_df["reward"].mean(),
        median_reward=full_df["reward"].median(),
        sum_reward=full_df["reward"].sum(),
        mean_reward_flat=full_df["reward_flat"].mean(),
        median_reward_flat=full_df["reward_flat"].median(),
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