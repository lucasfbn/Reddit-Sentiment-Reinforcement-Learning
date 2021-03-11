import pandas as pd
import pickle as pkl
import json
import paths
from utils import tracker


class Preprocessor:
    min_len = None

    fn_initial = "report.csv"
    fn_merge_hype_price = "merged.pkl"
    fn_cleaned = "cleaned.pkl"
    fn_timeseries = "timeseries.pkl"

    use_price = "close"

    # Used in merge_hype_price
    cols_to_be_scaled_daywise = ['num_comments', "score", 'pos', 'compound', 'neu', 'neg', 'n_posts']
    # Used in cleaner
    cols_to_be_dropped = ["date_day", "open", "close", "high", "low", "adj_close", "volume", "date_weekday",
                          "start_timestamp", "end_timestamp"]
    # Used in timeseries generator
    cols_to_be_scaled = ['num_comments', "score", 'pos', 'compound', 'neu', 'neg', 'n_posts',
                         "rel_change", 'price_ts']
    metadata_cols = ["price", "tradeable", "date", "available"]

    source_path = None
    target_path = None
    settings = {"used_price": use_price}

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

    def save_settings(self, obj):
        temp = {}
        for key, value in obj.__dict__.items():
            if key == "scaler":
                temp[key] = [value.__class__.__name__]
                continue
            if key not in ["df", "grps", "data"]:
                temp[key] = [value]
        Preprocessor.settings.update(temp)

    def settings_to_file(self):

        existing_df = None
        try:
            existing_df = pd.read_csv(paths.tracking_path / "preprocessing.csv", sep=";")
        except FileNotFoundError:
            print("Error loading report.")

        Preprocessor.settings["path"] = str(Preprocessor.target_path.name)
        df = pd.DataFrame(Preprocessor.settings)

        if existing_df is None:
            df.to_csv(paths.tracking_path / "preprocessing.csv", index=False, sep=";")
        else:
            existing_df = existing_df.append(df)
            existing_df.to_csv(paths.tracking_path / "preprocessing.csv", index=False, sep=";")

    def save(self, data, fn):
        with open(Preprocessor.target_path / fn, "wb") as f:
            pkl.dump(data, f)
