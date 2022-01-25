from dataclasses import dataclass, asdict
from datetime import datetime

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
    date: datetime = None
    price: float = None
    price_raw: float = None
    ticker_name: str = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Eval:
    action: int = None
    hold_proba: float = None
    buy_proba: float = None
    sell_proba: float = None
    reward: float = None
    reward_backtracked: float = None
    days_cash_bound: int = None
    open_positions: int = None

    def split_probas(self, probas):
        self.hold_proba, self.buy_proba, self.sell_proba = probas

    def to_dict(self):
        return asdict(self)


@dataclass
class Portfolio:
    execute: bool = False

    def to_dict(self):
        return asdict(self)


class Sequence:

    def __init__(self, index: int = None, data: Data = None, metadata: Metadata = None, evl: Eval = None):
        self.index = index
        self.data = data
        self.metadata = metadata
        self.evl = Eval() if evl is None else evl
        self.portfolio = Portfolio()

    def drop_data(self):
        self.data = None

    def to_dict(self):
        return self.metadata.to_dict() | self.evl.to_dict()
