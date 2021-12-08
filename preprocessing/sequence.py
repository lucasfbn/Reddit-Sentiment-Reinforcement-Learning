from dataclasses import dataclass, asdict

import pandas as pd


@dataclass
class Data:
    df: pd.DataFrame = None
    flat: pd.DataFrame = None
    arr: pd.DataFrame = None


@dataclass
class Metadata:
    tradeable: bool = None
    available: bool = None
    sentiment_data_available: bool = None
    date: pd.datetime = None
    price: float = None
    price_raw: float = None
    ticker_name: str = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Eval:
    action: int = None
    probas: list = None
    hold_proba: float = None
    buy_proba: float = None
    sell_proba: float = None
    reward: float = None

    def __post_init__(self):
        self.hold_proba, self.buy_proba, self.sell_proba = self.probas

    def to_dict(self):
        return asdict(self)


class Sequence:

    def __init__(self, index: int = None, data: Data = None, metadata: Metadata = None, evl: Eval = None):
        self.index = index
        self.data = data
        self.metadata = metadata
        self.evl = evl

    def drop_data(self):
        self.data = None
