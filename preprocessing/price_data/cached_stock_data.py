import pandas as pd
import datetime
from preprocessing.price_data.cache import Cache
from preprocessing.price_data.stock_prices import StockPrices

from more_itertools import consecutive_groups


class CachedStockData:
    date_col = "date_day"

    def __init__(self, ticker, start_date, end_date, live):
        self.live = live
        self.end_date = end_date
        self.start_date = start_date
        self.ticker = ticker

        self.c = None

    def initialize_cache(self, db_path=None):
        self.c = Cache() if db_path is None else Cache(db_path)

    def get_from_cache(self):
        return self.c.get(self.ticker)

    def get_last_date(self, df):
        return df[self.date_col].max()

    def download(self, start_date, end_date):
        sp = StockPrices(ticker_name=self.ticker, start_date=start_date, end_date=end_date, live=self.live)
        data = sp.download()
        data["ticker"] = self.ticker
        return data

    def filter_timespan(self, df):
        # TODO Just order and select per index?
        df = df[(df[self.date_col] >= self.start_date) & (df[self.date_col] <= self.end_date)]
        return df

    def get(self):
        df = self.get_from_cache()
        last_date = self.get_last_date(df)
        if last_date < self.end_date:
            new_df = self.download(start_date=last_date + datetime.timedelta(days=1),
                                   end_date=self.end_date)
            self.c.append(new_df, drop_duplicates=False)
            df = self.get_from_cache()

        df = self.filter_timespan(df)
        return df.reset_index(drop=True)
