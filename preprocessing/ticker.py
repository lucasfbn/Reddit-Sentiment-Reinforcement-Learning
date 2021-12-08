from dataclasses import dataclass, asdict

import pandas as pd


@dataclass
class Eval:
    reward: float = None
    open_positions: int = None
    min_date = pd.datetime = None
    max_date = pd.datetime = None

    def to_dict(self):
        return asdict(self)


class Ticker:

    def __init__(self, name, df=None):
        self.df = df
        self.name = name
        self.exclude = False

        self.sequences = []
        self.eval = Eval()

    def drop_data(self):
        _ = [seq.drop_data() for seq in self.sequences]

    def __len__(self):
        return len(self.sequences)
