import pickle as pkl

import pandas as pd

from utils.mlflow_api import log_file


class Preprocessor:
    min_len = None

    fn_initial = "report.csv"
    fn_merge_hype_price = "merged.pkl"
    fn_cleaned = "cleaned.pkl"
    fn_timeseries = "timeseries.pkl"

    source_path = None
    target_path = None

    def fix_cols(self, cols):
        new_cols = []
        for col in cols:
            col = col.lower()
            col = col.replace(" #", "")
            col = col.strip()
            col = col.replace(" ", "_")
            new_cols.append(col)
        return new_cols

    def load(self, fn, initial=False):

        path = Preprocessor.target_path
        if initial:
            path = Preprocessor.source_path

        if "csv" in fn:
            return pd.read_csv(path / fn, sep=";")
        elif "pkl" in fn:
            with open(path / fn, "rb") as f:
                return pkl.load(f)
        else:
            raise ValueError("Invalid filename.")

    def save(self, data, fn):
        assert fn is not None
        log_file(data, fn)
