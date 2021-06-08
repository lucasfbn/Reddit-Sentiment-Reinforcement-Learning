import datetime, sys, os

import yfinance as yf
import pandas as pd

import paths


class MissingDataException(Exception):
    def __init__(self, ticker, empty_df="not specified"):
        super().__init__(f"Couldn't retrieve prices for ticker '{ticker}'. Empty DataFrame: {empty_df}."
                         f" Check if markets are open.")


class OldDataException(Exception):
    def __init__(self, last_date, ticker):
        super().__init__(f"Couldn't retrieve most recent data. Last date retrieved for ticker '{ticker}': {last_date}")


class StockPrices:

    def __init__(self, ticker_name: str, ticker_df: pd.DataFrame, start_offset: int, live: bool):
        self.ticker = ticker_name
        self.df = ticker_df

        self.start_offset = start_offset
        self.live = live

        self.prices = None

    def _get_prices(self, start, end, interval="1d"):
        start, end = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d") if end is not None else None

        # Disable print statements because yfinance prints an annoying error when downloading fails.
        # I handle those errors myself (with a proper exception)
        sys.stdout = open(os.devnull, 'w')
        df = yf.download(self.ticker, start=start, end=end, interval=interval, progress=False)
        sys.stdout = sys.__stdout__  # enable print statements

        return df

    def _get_min_max_date(self):
        min_ = self.df["date_day"].min() - datetime.timedelta(days=self.start_offset)
        max_ = self.df["date_day"].max() + datetime.timedelta(days=1)
        return min_, max_

    def _live_assertions(self, prices):
        current_price = current_price = prices.tail(1)

        prices = prices.copy()
        prices = prices.drop(prices.tail(1).index)

        end = datetime.datetime.now()

        # Check if any data is missing
        if prices.empty or current_price.empty:
            raise MissingDataException(ticker=self.ticker)

        # Checks the following:
        # - whether the last historic date is not the current date
        # - whether the last historic date and the current date is not the same
        # - whether the current date is the current date
        last_current_date = current_price.tail(1).index.to_pydatetime()[0].date()
        last_historic_date = prices.tail(1).index.to_pydatetime()[0].date()

        if not (last_historic_date != end.date() and
                last_current_date != last_historic_date and
                last_current_date == end.date()):
            raise OldDataException(ticker=self.ticker, last_date=str(last_current_date))

    def _get_live_data(self):
        start, _ = self._get_min_max_date()

        # Retrieve full price data
        prices = self._get_prices(start, end=None)

        # Run assertions
        self._live_assertions(prices)
        return prices

    def _get_historic_data(self):
        start, end = self._get_min_max_date()
        prices = self._get_prices(start - datetime.timedelta(days=self.start_offset), end)

        if prices.empty:
            raise MissingDataException(ticker=self.ticker)

        return prices

    def download(self):

        if self.live:
            self.prices = self._get_live_data()
        else:
            self.prices = self._get_historic_data()

        # Add date_day column from index
        self.prices["date_day"] = pd.to_datetime(self.prices.index).to_period('D')
        # Index is the dates, we want a numerical index
        self.prices = self.prices.reset_index(drop=True)

        return self.prices

    def merge(self):
        """
        Merges the price data with the original df. Be aware that is merges the "outer" values, meaning that gaps
        between two dates will be filled with price data (as long as there is any at the specific day).
        """
        return self.prices.merge(self.df, on="date_day", how="outer")


class IndexPerformance:

    def __init__(self, min_date, max_date):
        self.max_date = max_date
        self.min_date = min_date

        self.performance = None

    def _calc_performance(self, df):
        start = df["Close"].iloc[0]
        end = df["Close"].iloc[len(df) - 1]

        df["pct"] = df["Close"] / start
        df = df[["Close", "pct"]]

        perf = end / start
        self.performance = {"start": start, "end": end, "perf": perf, "perf_detailed": df}

    def calc_index_comparison(self):
        df = yf.download("^GSPC", start=self.min_date.strftime("%Y-%m-%d"),
                         end=self.max_date.strftime("%Y-%m-%d"), interval="1d")
        self._calc_performance(df)

    def get_index_comparison(self):
        return self.performance