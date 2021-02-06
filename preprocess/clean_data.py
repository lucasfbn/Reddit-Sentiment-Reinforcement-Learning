import paths
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler


def drop_short(data, min_len=7, keep_offset=2):
    new_grps = []

    for grp in data:

        df = grp["data"]
        temp = df.dropna()

        if len(temp) >= min_len:
            min_index = len(df) - len(temp) - keep_offset
            grp["data"] = df.iloc[min_index:]
            new_grps.append(grp)

    return new_grps


def drop_yahoo_all_nan(data):
    new_grps = []
    for grp in data:
        if not grp["data"]["Close"].isnull().all():
            new_grps.append(grp)

    return new_grps


def drop_unnecessary(data, cols=["date_day", "Open", "High", "Low", "Adj Close", "Volume", "change_hype_level"]):
    new_grps = []

    for grp in data:
        grp["data"] = grp["data"].drop(columns=cols)
        new_grps.append(grp)

    return new_grps


def fill_nan(data):
    diff = data[0]["data"].columns.difference(["Close"])

    for grp in data:
        df = grp["data"]
        df[diff] = df[diff].fillna(0)
        grp["data"] = df
    return data


def drop_nan(data):
    for grp in data:
        grp["data"]["Close"] = grp["data"]["Close"].dropna()

    return data


def min_max_scaler(data):
    scaler = MinMaxScaler(feature_range=(0, 1))

    cols = data[0]["data"].columns

    for grp in data:
        # We want to keep a column with the unscaled price
        price_raw = grp["data"]["Close"].reset_index(drop=True)

        grp["data"] = pd.DataFrame(scaler.fit_transform(grp["data"]))
        grp["data"].columns = cols
        grp["data"]["price_raw"] = price_raw

    return data


with open(paths.data_path / "data_offset.pkl", "rb") as f:
    data = pkl.load(f)

data = drop_short(data)
print(len(data))
data = drop_yahoo_all_nan(data)
print(len(data))
data = drop_unnecessary(data)
data = fill_nan(data)
data = drop_nan(data)
data = min_max_scaler(data)
with open(paths.data_path / "data_cleaned.pkl", "wb") as f:
    pkl.dump(data, f)
