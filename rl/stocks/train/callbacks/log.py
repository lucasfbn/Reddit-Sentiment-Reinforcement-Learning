from rl.common.callbacks.logger import BaseLogCallback


class LogCallback(BaseLogCallback):

    def metrics(self, full_df, episode_end_df):
        return dict(
            sum=full_df["reward"].sum(),
            even=sum(full_df["reward"] == 0.0) / len(full_df),
            pos=sum(full_df["reward"] > 0.0) / len(full_df),
            neg=sum(full_df["reward"] < 0.0) / len(full_df)
        )
