from dataclasses import dataclass, asdict

from datetime import datetime
import pandas as pd


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

    @property
    def min_max_dates(self):
        min_ = self.lst[0].metadata.date
        max_ = self.lst[len(self.lst) - 1].metadata.date
        return min_, max_

    @property
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

    def __iter__(self):
        return iter(self.lst)


class Ticker:

    def __init__(self, name, df=None):
        self.df = df
        self.name = name
        self.exclude = False

        self.sequences = Sequences()
        self.evl = Eval()

    def __len__(self):
        return len(self.sequences.lst)
