from preprocessing.preprocessing_utils.preprocessor import Preprocessor
from utils import tracker


class Cleaner(Preprocessor):

    def __init__(self, keep_offset):
        self.data = self.load(self.fn_merge_hype_price)
        self.keep_offset = keep_offset

    def _rename_cols(self):
        for grp in self.data:
            grp["data"].columns = self.fix_cols(grp["data"].columns)

    def _assign_price_col(self):
        for grp in self.data:
            grp["data"]["price"] = grp["data"][self.use_price]

    def _mark_tradeable(self):
        for grp in self.data:
            df = grp["data"]
            df["date_weekday"] = df["date"].dt.dayofweek
            df["tradeable"] = df["date_weekday"] < 5

    def _forward_fill(self):
        for grp in self.data:
            grp["data"]["price"] = grp["data"]["price"].fillna(method="ffill")

    def _drop_short(self):
        new_data = []

        for grp in self.data:

            df = grp["data"]
            temp = df.dropna()

            if len(temp) >= self.min_len and len(df) >= len(temp) + self.keep_offset:
                min_index = len(df) - len(temp) - self.keep_offset
                grp["data"] = df.iloc[min_index:]
                new_data.append(grp)

        self.data = new_data

    def _drop_price_all_nan(self):
        new_data = []
        for grp in self.data:
            if not grp["data"]["price"].isnull().all():
                new_data.append(grp)
        self.data = new_data

    def _drop_unnecessary(self):
        for grp in self.data:
            grp["data"] = grp["data"].drop(columns=self.cols_to_be_dropped)

    def _fill_nan(self):
        diff = self.data[0]["data"].columns.difference(["price"])

        for grp in self.data:
            df = grp["data"]
            df[diff] = df[diff].fillna(0)
            grp["data"] = df

    def _drop_nan(self):
        for grp in self.data:
            grp["data"]["price"] = grp["data"]["price"].dropna()

    def pipeline(self):
        self._rename_cols()
        self._assign_price_col()
        self._mark_tradeable()
        self._forward_fill()
        self._drop_unnecessary()
        self._drop_short()
        self._drop_price_all_nan()
        self._fill_nan()
        self._drop_nan()
        self.save(self.data, self.fn_cleaned)
