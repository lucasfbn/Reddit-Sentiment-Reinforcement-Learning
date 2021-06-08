import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from preprocessing.preprocessing_utils.preprocessor import Preprocessor
from utils.utils import log


class TimeseriesGenerator(Preprocessor):
    fn = None

    def __init__(self, look_back, metadata_cols, check_availability, scale=True, scaler=MinMaxScaler(),
                 keep_unscaled=False, live=False):
        self.data = self.load(self.fn_cleaned)
        self.look_back = look_back
        self.metadata_cols = metadata_cols
        self.check_availability = check_availability
        self.scale = scale
        self.scaler = scaler
        self.keep_unscaled = keep_unscaled
        self.live = live

    @staticmethod
    def add_timeseries_price_col(grp):
        grp["data"]["price_ts"] = grp["data"]["price"]
        return grp

    @staticmethod
    def add_relative_change(grp):
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

    @staticmethod
    def reorder_cols(grp):
        # Make sure price col is the last col
        df = grp["data"]
        cols = list(df.columns)
        cols.remove("price_ts")
        cols = cols + ["price_ts"]
        df = df[cols]
        grp["data"] = df
        return grp

    def enforce_min_len(self, grp):
        if grp is None or len(grp["data"]) < 2:
            return False
        return True

    def pipeline(self):
        log.info("Running timeseries generator...")
        processed_data = []
        for grp in tqdm(self.data):
            grp = self.add_timeseries_price_col(grp)
            grp = self.add_relative_change(grp)

            grp = self.reorder_cols(grp)
            grp = self.make_sequence(grp)

            if not self.enforce_min_len(grp):
                continue

            grp = self.extract_metadata(grp)
            grp = self.apply_scaling(grp)
            grp = self.to_list(grp)

            processed_data.append(grp)

        self.data = processed_data
        self.save(self.data, self.fn)
        return self.data

    def apply_scaling(self, grp):
        raise NotImplemented

    def extract_metadata(self, grp):
        raise NotImplemented

    def make_sequence(self, grp):
        raise NotImplemented

    def to_list(self, grp):
        raise NotImplemented


class TimeseriesGeneratorNN(TimeseriesGenerator):
    fn = "nn_input.pkl"

    def make_sequence(self, grp):
        df = grp["data"]

        shifted = []

        for lb in range(1, self.look_back + 1):
            shifted.append(df.shift(-lb))

        suffix_counter = 0
        for i in range(len(shifted)):
            df = df.join(shifted[i], lsuffix=f"_{suffix_counter}", rsuffix=f"")
            suffix_counter += 2

        df = df.dropna()

        if self.check_availability:
            # 'available' without suffix is the the column of the actual day and therefore relevant for the available
            # filter
            df = df[df["available"] == True]

        if len(df) == 0:
            return None

        grp["data"] = df
        return grp

    def apply_scaling(self, grp):

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

    def extract_metadata(self, grp):
        grp["metadata"] = grp["data"][self.metadata_cols]
        grp["data"] = grp["data"].drop(columns=self.metadata_cols)

        grp = self._del_metadata_from_data(grp)
        return grp

    def _del_metadata_from_data(self, grp):
        df = grp["data"]

        new_cols = []
        for df_col in df.columns:
            df_col_raw = re.sub("_\d+", "", df_col)  # Gets rid of _n. For instance: score_0 -> score
            if df_col_raw not in self.metadata_cols:
                new_cols.append(df_col)

        grp["data"] = grp["data"][new_cols]
        return grp

    def to_list(self, grp):
        df = grp["data"]
        df = df.to_dict(orient="records")
        new_grp = []
        for timeseries in df:
            new_grp.append(pd.DataFrame([timeseries]))
        grp["data"] = new_grp
        return grp


class TimeseriesGeneratorCNN(TimeseriesGenerator):
    fn = "cnn_input.pkl"

    def apply_scaling(self, grp):
        if not self.scale:
            return grp

        cols_to_be_scaled = [col for col in grp["data"][0] if "unscaled" not in col]

        for i, df in enumerate(grp["data"]):
            cols = df.columns

            ct = ColumnTransformer(
                [("_", self.scaler, cols_to_be_scaled)], remainder="passthrough"
            )
            df = ct.fit_transform(df)
            grp["data"][i] = pd.DataFrame(df, columns=cols)
        return grp

    def make_sequence(self, grp):
        df = grp["data"]
        sequences = []

        for i in range(len(df)):

            if self.check_availability:
                available = df["available"].iloc[i]
                if not available or (i - self.look_back < 0):
                    continue
                else:
                    sequences.append(df[i - self.look_back:i+1])
            else:
                if i > (len(df) - self.look_back):
                    break
                sequences.append(df[i:i + self.look_back])

        assert len(sequences) > 0
        grp["data"] = sequences
        return grp

    def extract_metadata(self, grp):
        metadata = []
        for i, seq in enumerate(grp["data"]):
            metadata.append(seq.tail(1)[self.metadata_cols])
            grp["data"][i] = grp["data"][i].drop(columns=self.metadata_cols)
        grp["metadata"] = pd.concat(metadata)
        return grp

    def to_list(self, grp):
        return grp


class TimeseriesGeneratorWrapper:

    def __init__(self, **kwargs):
        self.tsg_nn = TimeseriesGeneratorNN(**kwargs)
        self.tsg_cnn = TimeseriesGeneratorCNN(**kwargs)

    def pipeline(self):
        self.tsg_nn.pipeline()
        self.tsg_cnn.pipeline()
