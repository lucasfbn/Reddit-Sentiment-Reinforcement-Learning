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

    def generate_date_range(self):
        return pd.date_range(self.start_date.to_timestamp(),
                             self.end_date.to_timestamp(), freq="D")

    def get_missing_dates(self, df, date_rng):
        date_range = pd.DataFrame()
        date_range[self.date_col] = date_rng
        date_range[self.date_col] = date_range[self.date_col].dt.to_period("D")

        merged = date_range.merge(df, how="outer", indicator=True)
        missing = merged[merged["_merge"] == "left_only"]
        missing = missing.drop(columns=["_merge"])
        missing = missing.reset_index(drop=True)

        return missing

    def filter_weekends(self, df):
        df["weekend"] = df[self.date_col].dt.weekday
        df = df[df["weekend"] < 5]
        df = df.drop(columns=["weekend"])
        return df

    def get_consecutive_sequences(self, df):
        df = df.sort_values(by=[self.date_col])
        dates = df[self.date_col].dt.to_timestamp().tolist()
        return [list(cg) for cg in consecutive_groups(dates, lambda x: x.toordinal())]

    def get_from_cache(self):
        df = self.c.get(self.ticker)
        # TODO Just order and select per index?
        df = df[(df[self.date_col] >= self.start_date) & (df[self.date_col] <= self.end_date)]
        return df

    def download(self, date_sequences):
        for ds in date_sequences:
            sp = StockPrices(ticker_name=self.ticker, start_date=ds[0],
                             end_date=ds[1], live=self.live)
            data = sp.download()
            data["ticker"] = self.ticker
            self.c.append(data, drop_duplicates=False)

        self.c.drop_duplicates()

    def get(self):
        rng = self.generate_date_range()
        cache_df = self.get_from_cache()
        missing_dates = self.get_missing_dates(cache_df, rng)
        missing_dates = self.filter_weekends(missing_dates)
        missing_dates = self.get_consecutive_sequences(missing_dates)
        self.download(missing_dates)
        return self.get_from_cache()
