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


def scale_price(data):
    # ONLY scales close price

    for grp in data:
        scaler = MinMaxScaler()
        df = grp["data"]
        df["close_scaled"] = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
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
# data = scale_price(data)
with open(paths.data_path / "data_cleaned.pkl", "wb") as f:
    pkl.dump(data, f)
