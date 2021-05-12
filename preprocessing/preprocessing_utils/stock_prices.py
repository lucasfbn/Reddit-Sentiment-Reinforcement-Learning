import datetime

import requests_cache
import yfinance as yf

import paths

requests_cache.install_cache(cache_name=paths.stock_cache, backend="sqlite")


class StockPrices:

    def __init__(self, grp, start_offset, live=False):
        self.ticker = grp["ticker"]
        self.data = grp["data"]

        self.start_offset = start_offset
        self.live = live

    def _get_prices(self, start, end, interval="1d"):
        start, end = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        df = yf.download(self.ticker, start=start, end=end, interval=interval, progress=False)
        return df

    def _get_min_max_date(self):
        min_ = self.data["date_day"].min() - datetime.timedelta(days=self.start_offset)
        max_ = self.data["date_day"].max()
        return min_, max_

    def _get_live(self):
        start, _ = self._get_min_max_date()
        end = datetime.datetime.now()

        current_price = self._get_prices(end, end + datetime.timedelta(days=1), interval="1m").tail(1).tz_localize(None)
        historic = self._get_prices(start, end)

        merged = historic.append(current_price)
        merged["date_day"] = pd.to_datetime(merged.index).to_period('D')
        merged = merged.reset_index(drop=True)

        if not historic.empty:
            last_historic_date = historic.tail(1).index.to_pydatetime()[0].date()
            assert last_historic_date != end.date()

        return merged

    def _get_historic(self):
        start, end = self._get_min_max_date()
        historic = self._get_prices(start - datetime.timedelta(days=self.start_offset), end)
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


if __name__ == "__main__":
    import pandas as pd

    sp = StockPrices(data[0], 10, True)
    df = sp.download()
    # print()
    pass
