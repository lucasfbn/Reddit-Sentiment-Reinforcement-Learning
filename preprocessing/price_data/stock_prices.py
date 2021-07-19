import datetime
import os
import sys

import pandas as pd
import yfinance as yf


class MissingDataException(Exception):
    def __init__(self, ticker, empty_df="not specified"):
        super().__init__(f"Couldn't retrieve prices for ticker '{ticker}'. Empty DataFrame: {empty_df}."
                         f" Check if markets are open.")


class OldDataException(Exception):
    def __init__(self, last_date, ticker):
        super().__init__(f"Couldn't retrieve most recent data. Last date retrieved for ticker '{ticker}': {last_date}")


class StockPrices:

    date_col = "date_day_shifted"

    def __init__(self, ticker_name: str, start_date: pd.Period, end_date: pd.Period, live: bool):
        self.ticker = ticker_name
        self.start_date = start_date
        self.end_date = end_date
        self.live = live

        self.prices = None

        self._correct_dates()

    def _correct_dates(self):
        """
        yfinance specific correction such that the correct dates are downloaded.
        """
        self.start_date = self.start_date + datetime.timedelta(days=1)
        self.end_date = self.end_date + datetime.timedelta(days=1)

    def _get_prices(self, start, end, interval="1d"):
        start, end = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d") if end is not None else None

        # Disable print statements because yfinance prints an annoying error when downloading fails.
        # I handle those errors myself (with a proper exception)
        sys.stdout = open(os.devnull, 'w')
        df = yf.download(self.ticker, start=start, end=end, interval=interval, progress=False)
        sys.stdout = sys.__stdout__  # enable print statements

        return df

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
        # Retrieve full price data
        prices = self._get_prices(self.start_date, end=None)

        # Run assertions
        self._live_assertions(prices)
        return prices

    def _get_historic_data(self):
        prices = self._get_prices(self.start_date, self.end_date)

        if prices.empty:
            raise MissingDataException(ticker=self.ticker)

        return prices

    def _remove_spaces_from_cols(self):
        self.prices.columns = [col.replace(" ", "_") for col in self.prices.columns]

    def download(self):

        if self.live:
            self.prices = self._get_live_data()
        else:
            self.prices = self._get_historic_data()

        # Add date_day column from index
        self.prices[self.date_col] = pd.to_datetime(self.prices.index).to_period('D')
        # Index is the dates, we want a numerical index
        self.prices = self.prices.reset_index(drop=True)

        self._remove_spaces_from_cols()
        return self.prices


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
