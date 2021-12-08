from dataclasses import dataclass, asdict

import pandas as pd


@dataclass
class Data:
    flat = pd.DataFrame
    arr = pd.DataFrame


@dataclass
class Metadata:
    ticker_name: str
    tradeable: bool
    available: bool
    sentiment_data_available: bool
    date: pd.datetime
    price: float
    price_raw: float

    def to_dict(self):
        return asdict(self)


@dataclass
class Eval:
    action: int
    probas: list
    hold_proba: float = None
    buy_proba: float = None
    sell_proba: float = None
    reward: float = None

    def __post_init__(self):
        self.hold_proba, self.buy_proba, self.sell_proba = self.probas

    def to_dict(self):
        return asdict(self)


class Sequence:

    def __init__(self, index: int = None, data: Data = None, metadata: Metadata = None, eval: Eval = None):
        self.index = index
        self.data = data
        self.metadata = metadata
        self.eval = eval

    def drop_data(self):
        self.data = None
