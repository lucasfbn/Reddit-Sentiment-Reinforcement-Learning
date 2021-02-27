import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from preprocess.preprocessor import Preprocessor
from utils import tracker


class TimeseriesGenerator(Preprocessor):

    def __init__(self, look_back, scaler=MinMaxScaler(), live=False):
        self.data = self.load(self.fn_cleaned)
        self.look_back = look_back
        self.scaler = scaler
        self.live = live

        tracker.add({"look_back": self.look_back,
                     "scaler": self.scaler.__class__.__name__}, "TimeseriesGenerator")

    def _add_timeseries_price_col(self, grp):
        grp["data"]["price_ts"] = grp["data"]["price"]
        return grp

    def _add_relative_change(self, grp):
        df = grp["data"]
        prices = df["price"].values.tolist()

        rel_change = []
        for i, _ in enumerate(prices):
            if i == 0:
                rel_change.append(0.0)
            else:
                rel_change.append((prices[i] - prices[i - 1]) / (prices[i - 1]))

        grp["data"]["rel_change"] = rel_change
        return grp

    def _add_pre_data(self, grp):
        df = grp["data"]

        shifted = []

        for lb in range(1, self.look_back + 1):
            shifted.append(df.shift(-lb))

        suffix_counter = 0
        for i in range(len(shifted)):
            df = df.join(shifted[i], lsuffix=f"_{suffix_counter}", rsuffix=f"")
            suffix_counter += 2

        df = df.dropna()
        grp["data"] = df
        return grp

    def _scale(self, grp):
        df = grp["data"]
        price = df["price"].reset_index(drop=True)
        tradeable = df["tradeable"].reset_index(drop=True)
        date = df["date"].reset_index(drop=True)

        new_df = pd.DataFrame()

        for col in self.cols_to_be_scaled:
            all_cols = []
            for df_col in df.columns:
                df_col_raw = df_col.rsplit('_', 1)[0] # Gets rid of _n. For instance: score_0 -> score
                if col == df_col_raw:
                    all_cols.append(df_col)

            temp = df[all_cols]
            temp = temp.values.T  # Transpose so we have each row as column
            temp = self.scaler.fit_transform(temp)
            temp = pd.DataFrame(temp.T, columns=all_cols)  # Transpose back
            new_df = new_df.merge(temp, how="right", left_index=True, right_index=True)

        new_df["price"] = price
        new_df["tradeable"] = tradeable
        new_df["date"] = date
        grp["data"] = new_df
        return grp

    def _live(self, grp):
        if self.live:
            grp["data"] = grp["data"].tail(1)
        return grp

    def pipeline(self):
        processed_data = []
        for grp in self.data:
            grp = self._add_timeseries_price_col(grp)
            grp = self._add_relative_change(grp)
            grp = self._add_pre_data(grp)

            if len(grp["data"]) == 0:
                continue

            grp = self._scale(grp)
            grp = self._live(grp)
            processed_data.append(grp)

        self.data = processed_data
        self.save(self.data, self.fn_timeseries)
