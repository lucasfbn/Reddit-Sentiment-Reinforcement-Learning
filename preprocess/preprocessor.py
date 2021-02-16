import pandas as pd
import pickle as pkl
import json


class Preprocessor:
    fn_initial = "report.csv"
    fn_merge_hype_price = "merged.pkl"
    fn_cleaned = "cleaned.pkl"
    fn_timeseries = "timeseries.pkl"

    use_price = "close"

    # Used in cleaner
    cols_to_be_dropped = ["date_day", "open", "close", "high", "low", "adj_close", "volume", "date_weekday"]
    # Used in timeseries generator
    cols_to_be_scaled = ['total_hype_level', 'current_hype_level', 'previous_hype_level',
                         'posts', 'upvotes', 'comments', 'distinct_authors', "change_hype_level", "rel_change",
                         'price_ts']

    path = None
    settings = []

    def fix_cols(self, cols):
        new_cols = []
        for col in cols:
            col = col.lower()
            col = col.replace(" #", "")
            col = col.strip()
            col = col.replace(" ", "_")
            new_cols.append(col)
        return new_cols

    def load(self, fn):
        if "csv" in fn:
            return pd.read_csv(Preprocessor.path / fn, sep=",")
        elif "pkl" in fn:
            with open(Preprocessor.path / fn, "rb") as f:
                return pkl.load(f)
        else:
            raise ValueError("Invalid filename.")

    def save_settings(self, obj):
        obj_name = obj.__class__.__name__
        temp = {obj_name: []}
        for key, value in obj.__dict__.items():
            if key == "scaler":
                temp[obj_name].append({key: value.__class__.__name__})
                continue
            if key not in ["df", "grps", "data"]:
                temp[obj_name].append({key: value})
        Preprocessor.settings.append(temp)

    def settings_to_file(self):
        with open(Preprocessor.path / "settings.json", "w+") as fp:
            json.dump(Preprocessor.settings, fp)

    def save(self, data, fn):
        with open(Preprocessor.path / fn, "wb") as f:
            pkl.dump(data, f)
