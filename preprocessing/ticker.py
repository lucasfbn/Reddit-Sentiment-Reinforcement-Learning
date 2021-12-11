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


class Ticker:

    def __init__(self, name, df=None):
        self.df = df
        self.name = name
        self.exclude = False

        self.sequences = []
        self.evl = Eval()

    def set_min_max_date(self):
        self.evl.min_date = self.sequences[0].metadata.date
        self.evl.max_date = self.sequences[len(self.sequences) - 1].metadata.date

    def drop_sequence_data(self):
        _ = [seq.drop_data() for seq in self.sequences]

    def aggregate_rewards(self):
        self.evl.reward = sum(seq.evl.reward for seq in self.sequences)
        return self.evl.reward

    def aggregate_sequence_eval(self):
        return pd.DataFrame([seq.evl.to_dict() for seq in self.sequences])

    def __len__(self):
        return len(self.sequences)
