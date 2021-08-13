import datetime

from pandas import Period

from preprocessing.price_data.cache import Cache
from preprocessing.price_data.stock_prices import StockPrices
from utils.util_funcs import log


class CachedStockData:
    date_col = "date_day_shifted"

    standard_start = Period('2021-02-01', 'D')

    def __init__(self, ticker, start_date, end_date, live):
        self.live = live
        self.end_date = self._get_end_date(end_date, live)
        self.start_date = start_date
        self.ticker = ticker

        self.c = None

    def _get_end_date(self, end_date, live):
        return end_date if not live else Period.now("D")

    def initialize_cache(self, db_path=None):
        self.c = Cache() if db_path is None else Cache(db_path)

    def get_from_cache(self):
        return self.c.get(self.ticker)

    def get_last_date(self, df):
        if df.empty:
            return None
        return df[self.date_col].max()

    def download(self, start_date, end_date):
        sp = StockPrices(ticker_name=self.ticker, start_date=start_date, end_date=end_date, live=self.live)
        data = sp.download()
        data["ticker"] = self.ticker
        return data

    def filter_timespan(self, df):

        # Do not filter when live since we want the most recent data
        if self.live:
            return df[(df[self.date_col] >= self.start_date)]

        # TODO Just order and select per index?
        return df[(df[self.date_col] >= self.start_date) & (df[self.date_col] <= self.end_date)]

    def drop_duplicates(self, df):
        old_len = len(df)
        df = df.drop_duplicates(subset=[self.date_col], keep="last")

        if not old_len == len(df):
            log.warn(f"There are duplicates in the DB for ticker {self.ticker}")

        return df

    def get(self):
        df = self.get_from_cache()
        last_date = self.get_last_date(df)

        if last_date is None:
            new_df = self.download(start_date=self.standard_start,
                                   end_date=self.end_date)
            self.c.append(new_df, drop_duplicates=False)
            df = self.get_from_cache()
        elif last_date < self.end_date:
            new_df = self.download(start_date=last_date + datetime.timedelta(days=1),
                                   end_date=self.end_date)
            self.c.append(new_df, drop_duplicates=False)
            df = self.get_from_cache()

        df = self.drop_duplicates(df)
        df = self.filter_timespan(df)
        return df.reset_index(drop=True)
