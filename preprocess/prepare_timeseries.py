import pandas as pd
import paths
import pickle as pkl
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler


# Basically combines n rows into one row

def add_relative_change(data):
    for grp in data:
        df = grp["data"]
        prices = df["Close"].values.tolist()

        rel_change = []
        for i, _ in enumerate(prices):
            if i == 0:
                rel_change.append(0.0)
            else:
                rel_change.append((prices[i] - prices[i - 1]) / (prices[i - 1]))

        grp["data"]["rel_change"] = rel_change

    return data


def add_pre_data(data):
    for grp in data:
        df = grp["data"]
        shift_1 = df.shift(-1)
        shift_2 = df.shift(-2)

        join_1 = df.join(shift_1, lsuffix="_1", rsuffix="_2")
        join_2 = join_1.join(shift_2, rsuffix="_3")
        grp["data"] = join_2.dropna()
    return data


def scale(data, scaler):
    cols = ['total_hype_level', 'current_hype_level', 'previous_hype_level',
            'posts', 'upvotes', 'comments', 'distinct_authors',
            'rel_change']

    for grp in data:

        df = grp["data"]
        close = df["Close"].reset_index(drop=True)

        new_df = pd.DataFrame()

        for col in cols:
            all_cols = [col + "_1", col + "_2", col]

            temp = df[all_cols]
            temp = temp.values.T  # Transpose so we have each row as column
            temp = scaler.fit_transform(temp)
            temp = pd.DataFrame(temp.T, columns=all_cols)  # Transpose back
            new_df = new_df.merge(temp, how="right", left_index=True, right_index=True)

        new_df["Close"] = close
        grp["data"] = new_df

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
            df = df.reset_index(drop=True)
            grp["data"] = df

    for grp in data:
        assert len(grp["data"]) == max

    return data


with open(paths.data_path / "data_cleaned.pkl", "rb") as f:
    data = pkl.load(f)

data = add_relative_change(data)
data = add_pre_data(data)
data = scale(data, MaxAbsScaler())
data = add_padding(data)

with open(paths.data_path / "data_timeseries.pkl", "wb") as f:
    pkl.dump(data, f)
