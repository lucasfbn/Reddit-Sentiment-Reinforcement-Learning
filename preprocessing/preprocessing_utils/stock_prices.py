import datetime, sys, os

import requests_cache
import yfinance as yf
import pandas as pd

import paths

requests_cache.install_cache(cache_name=paths.stock_cache, backend="sqlite")


class MissingDataException(Exception):
    def __init__(self, ticker, empty_df="not specified"):
        super().__init__(f"Couldn't retrieve prices for ticker '{ticker}'. Empty DataFrame: {empty_df}."
                         f" Check if markets are open.")


class WrongDataException(Exception):
    def __init__(self, ticker):
        super().__init__(f"Current and historic data doesn't line up for ticker '{ticker}'")


class StockPrices:

    def __init__(self, grp, start_offset, live=False):
        self.ticker = grp["ticker"]
        self.data = grp["data"]

        self.start_offset = start_offset
        self.live = live

    def _get_prices(self, start, end, interval="1d"):
        start, end = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        # Disable print statements because yfinance prints an annoying error when downloading fails.
        # I handle those errors myself (with a proper exception)
        sys.stdout = open(os.devnull, 'w')
        df = yf.download(self.ticker, start=start, end=end, interval=interval, progress=False)
        sys.stdout = sys.__stdout__  # enable print statements
        
        return df

    def _get_min_max_date(self):
        min_ = self.data["date_day"].min() - datetime.timedelta(days=self.start_offset)
        max_ = self.data["date_day"].max()
        return min_, max_

    def _get_live(self):
        start, _ = self._get_min_max_date()
        end = datetime.datetime.now()

        current = self._get_prices(end, end + datetime.timedelta(days=1), interval="1m").tail(1).tz_localize(None)
        historic = self._get_prices(start, end)

        if historic.empty or current.empty:
            raise MissingDataException(ticker=self.ticker)

        last_current_date = current.tail(1).index.to_pydatetime()[0].date()
        last_historic_date = historic.tail(1).index.to_pydatetime()[0].date()

        if not (last_historic_date != end.date() and
                last_current_date != last_historic_date and
                last_current_date == end.date()):
            raise WrongDataException(ticker=self.ticker)

        merged = historic.append(current)
        merged["date_day"] = pd.to_datetime(merged.index).to_period('D')
        merged = merged.reset_index(drop=True)
        return merged

    def _get_historic(self):
        start, end = self._get_min_max_date()
        historic = self._get_prices(start - datetime.timedelta(days=self.start_offset), end)

        if historic.empty:
            raise MissingDataException(ticker=self.ticker)

        historic["date_day"] = pd.to_datetime(historic.index).to_period('D')
        return historic

    def download(self):

        if self.live:
            df = self._get_live()
        else:
            df = self._get_historic()

        merged = self._merge(df)
        return merged

    def _merge(self, historic):
        return historic.merge(self.data, on="date_day", how="outer")


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
