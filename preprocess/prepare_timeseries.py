import pandas as pd
import paths
import pickle as pkl
import numpy as np


# Basically combines n rows into one row

def add_pre_data(data):
    for grp in data:
        df = grp["data"]
        shift_1 = df.shift(-1)
        shift_2 = df.shift(-2)

        join_1 = df.join(shift_1, lsuffix="_1", rsuffix="_2")
        join_2 = join_1.join(shift_2, rsuffix="_3")
        grp["data"] = join_2.dropna()
    return data


def add_padding(data):
    cols = data[0]["data"].columns

    max = 0
    for grp in data:
        if len(grp["data"]) > max:
            max = len(grp["data"])

    for grp in data:

        df = grp["data"]
        diff = max - len(df)
        if diff != 0:
            first_row = df.values[0]
            first_row = np.tile(first_row, (diff, 1))
            temp_df = pd.DataFrame(first_row, columns=cols)
            df = pd.concat([temp_df, df])
            df = df.reset_index()  # The new row will have index of 0 aswell, so we have to reset it
            grp["data"] = df

    for grp in data:
        assert len(grp["data"]) == max

    return data


def drop_raw_price(data):
    for grp in data:
        grp["data"] = grp["data"].drop(columns=["price_raw_1", "price_raw_2"])
    return data


with open(paths.data_path / "data_cleaned.pkl", "rb") as f:
    data = pkl.load(f)

data = add_pre_data(data)
data = add_padding(data)
data = drop_raw_price(data)

with open(paths.data_path / "data_timeseries.pkl", "wb") as f:
    pkl.dump(data, f)
