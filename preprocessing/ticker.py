from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


@dataclass
class Eval:
    reward: float = None
    open_positions: int = None
    min_date: datetime = None
    max_date: datetime = None

    def to_dict(self):
        return asdict(self)


class Sequences:

    def __init__(self):
        self.lst = None

    def min_max_dates(self):
        min_ = self.lst[0].metadata.date
        max_ = self.lst[len(self.lst) - 1].metadata.date
        return min_, max_

    def aggregated_rewards(self):
        return sum(seq.evl.reward for seq in self.lst)

    def evl_to_df(self):
        return pd.DataFrame([seq.evl.to_dict() for seq in self.lst])

    def metadata_to_df(self):
        return pd.DataFrame([seq.metadata.to_dict() for seq in self.lst])

    def to_df(self):
        return pd.DataFrame([seq.evl.to_dict() | seq.metadata.to_dict() for seq in self.lst])

    def drop_data(self):
        _ = [seq.drop_data() for seq in self.lst]

    def backtrack(self):
        df = self.to_df()
        orig_df = df.copy()

        df = df[["action", "reward", "date"]]

        df = df[df["action"] != 0]
        df = df.query(
            "not(reward == 0.0 and action == 2)")  # or df = df[~((df["action"] == 2) & (df["reward"] == 0.0))]

        df["reward_shifted"] = df["reward"].shift(-1)
        df["reward_backtracked"] = (df["reward"] + df["reward_shifted"]).where(df["action"] == 1, np.nan)

        df["date_shifted"] = df["date"].shift(-1)
        df["timedelta"] = (
                pd.to_datetime(df["date_shifted"].astype(str)) - pd.to_datetime(df["date"].astype(str))).dt.days
        df["days_cash_bound"] = (df["timedelta"]).where(df["action"] == 1, np.nan)

        orig_df = orig_df.drop(columns=["reward_backtracked", "days_cash_bound"])
        orig_df = orig_df.join(df[["reward_backtracked", "days_cash_bound"]])
        orig_df = orig_df.replace([np.nan], [None])

        for index, row in orig_df.iterrows():
            self.lst[index].evl.reward_backtracked = row["reward_backtracked"]
            self.lst[index].evl.days_cash_bound = row["days_cash_bound"]

    def __iter__(self):
        return iter(self.lst)

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, item):
        return self.lst[item]


class Ticker:

    def __init__(self, name, df=None):
        self.df = df
        self.name = name
        self.exclude = False

        self.sequences = Sequences()
        self.evl = Eval()

    def __len__(self):
        return len(self.sequences.lst)
