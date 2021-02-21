import datetime
import pickle as pkl

import pandas as pd
import yfinance as yf

import paths
from preprocess.stock_prices import StockPrices
from preprocess.preprocessor import Preprocessor

pd.options.mode.chained_assignment = None


class MergeHypePrice(Preprocessor):

    def __init__(self,
                 start_hour=22, start_min=0,
                 market_symbols=[],
                 min_len_hype=7,
                 start_offset=30,
                 live=False,
                 limit=None):

        self.df = self.load(self.fn_initial)
        self.grps = []
        self.start_hour = start_hour
        self.start_min = start_min
        self.market_symbols = market_symbols
        self.min_len = min_len_hype
        self.start_offset = start_offset
        self.live = live
        self.limit = limit

    def _handle_time(self):
        self.df["time"] = pd.to_datetime(self.df["end"], format="%Y-%m-%d %H:%M")
        self.df = self.df.sort_values(by=["time"])
        self.df["time_shifted"] = self.df["time"] - pd.Timedelta(hours=self.start_hour,
                                                                 minutes=self.start_min) + pd.Timedelta(days=1)

        self.df["date_day"] = pd.to_datetime(self.df['time_shifted']).dt.to_period('D')

    def _filter_market_symbol(self):
        if self.market_symbols:
            self.df = self.df[self.df["market_symbol"].isin(self.market_symbols)]

    def _grp_by(self):
        grp_by = self.df.groupby(["ticker"])

        for name, group in grp_by:
            self.grps.append({"ticker": name, "data": group.groupby(["date_day"]).agg("sum").reset_index()})

    def _limit(self):
        if self.limit is not None:
            self.grps = self.grps[:self.limit]

    def _drop_short(self):
        filtered_grps = []

        for grp in self.grps:
            if len(grp["data"]) >= self.min_len:
                filtered_grps.append(grp)
        self.grps = filtered_grps

    def _add_stock_prices(self):
        new_grps = []

        for i, grp in enumerate(self.grps):
            print(f"Processing {i}/{len(self.grps)}")
            sp = StockPrices(grp, start_offset=self.start_offset, live=self.live)
            new_grps.append({"ticker": grp["ticker"], "data": sp.download()})

        self.grps = new_grps

    def pipeline(self):
        self._handle_time()
        self._filter_market_symbol()
        self._grp_by()
        self._drop_short()
        self._limit()
        self._add_stock_prices()
        self.save(self.grps, self.fn_merge_hype_price)
        self.save_settings(self)
