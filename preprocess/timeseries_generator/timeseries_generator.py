import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from preprocess.preprocessor import Preprocessor
from utils import tracker


class TimeseriesGenerator(Preprocessor):

    def __init__(self, look_back, scale=True, scaler=MinMaxScaler(), live=False):
        self.data = self.load(self.fn_cleaned)
        self.look_back = look_back
        self.scale = scale
        self.scaler = scaler
        self.live = live

        tracker.add({"look_back": self.look_back,
                     "scale": self.scale,
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

    def _reorder_cols(self, df):
        # Make sure price col is the last col
        cols = list(df.columns)
        cols.remove("price_ts")
        cols = cols + ["price_ts"]
        df = df[cols]
        return df

    def pipeline(self):
        processed_data = []
        for grp in self.data:
            grp = self._add_timeseries_price_col(grp)
            grp = self._add_relative_change(grp)
            # grp = self._extract_metadata(grp)
            grp = self._model_specific(grp)

            if grp is None:
                continue

            processed_data.append(grp)

        self.data = processed_data
        # self.save(self.data, self.fn_timeseries)
        return self.data


class TimeseriesGeneratorNN(TimeseriesGenerator):

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

        if not self.scale:
            return grp

        df = grp["data"]

        new_df = pd.DataFrame()

        # Get base cols (every unique col without are integer suffix)
        base_cols = []
        for col in df.columns:
            col = re.sub("_\d+", "", col)  # neu_0 -> neu, price_ts -> price_ts. Only matches "_01" etc.
            if col not in base_cols:
                base_cols.append(col)

        for col in base_cols:
            all_cols = []
            for df_col in df.columns:
                df_col_raw = re.sub("_\d+", "", df_col)  # Gets rid of _n. For instance: score_0 -> score
                if col == df_col_raw:
                    all_cols.append(df_col)

            temp = df[all_cols]
            temp = temp.values.T  # Transpose so we have each row as column
            temp = self.scaler.fit_transform(temp)
            temp = pd.DataFrame(temp.T, columns=all_cols)  # Transpose back
            new_df = new_df.merge(temp, how="right", left_index=True, right_index=True)

        grp["data"] = new_df
        return grp

    def _live(self, grp):
        if self.live:
            grp["data"] = grp["data"].tail(1)
        return grp

    def _extract_metadata(self, grp):
        grp["metadata"] = grp["data"][["price", "tradeable", "date"]]
        grp["data"] = grp["data"].drop(columns=["price", "tradeable", "date"])
        return grp

    def _del_metadata_from_data(self, grp):
        df = grp["data"]

        new_cols = []
        for df_col in df.columns:
            df_col_raw = re.sub("_\d+", "", df_col)  # Gets rid of _n. For instance: score_0 -> score
            if df_col_raw not in ["price", "tradeable", "date"]:
                new_cols.append(df_col)

        grp["data"] = grp["data"][new_cols]
        return grp

    def _model_specific(self, grp):

        tracker.add({"generator": self.__class__.__name__}, "TimeseriesGenerator")

        grp = self._add_pre_data(grp)
        grp = self._extract_metadata(grp)
        if len(grp["data"]) == 0:
            return None
        grp = self._del_metadata_from_data(grp)

        grp = self._scale(grp)
        grp["data"] = self._reorder_cols(grp["data"])
        grp = self._live(grp)
        return grp


class TimeseriesGeneratorCNN(TimeseriesGenerator):

    def _scale(self, grp):
        if not self.scale:
            return grp

        for i, df in enumerate(grp["data"]):
            cols = df.columns
            df = self.scaler.fit_transform(df)
            grp["data"][i] = pd.DataFrame(df, columns=cols)
        return grp

    def _make_seq(self, df):
        sequences = []

        for i in range(len(df)):
            if i > (len(df) - self.look_back):
                break
            sequences.append(df[i:i + self.look_back])
        return sequences

    def _extract_metadata(self, grp):
        metadata = []
        for i, seq in enumerate(grp["data"]):
            metadata.append(seq.tail(1)[["price", "tradeable", "date"]])
            grp["data"][i] = grp["data"][i].drop(columns=["price", "tradeable", "date"])
        grp["metadata"] = pd.concat(metadata)
        return grp

    def _model_specific(self, grp):

        tracker.add({"generator": self.__class__.__name__}, "TimeseriesGenerator")

        grp["data"] = self._reorder_cols(grp["data"])
        grp["data"] = self._make_seq(grp["data"])
        grp = self._extract_metadata(grp)
        grp = self._scale(grp)
        return grp
