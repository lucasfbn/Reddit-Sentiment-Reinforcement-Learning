import pickle as pkl

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import paths


# Basically combines n rows into one row

def add_price_col(data):
    for grp in data:
        grp["data"]["price"] = grp["data"]["Close"]
    return data


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


def add_pre_data(data, look_back=2):
    for grp in data:

        df = grp["data"]

        shifted = []

        for lb in range(1, look_back + 1):
            shifted.append(df.shift(-lb))

        suffix_counter = 0
        for i in range(len(shifted)):
            df = df.join(shifted[i], lsuffix=f"_{suffix_counter}", rsuffix=f"_{suffix_counter + 1}")
            suffix_counter += 2

        df = df.dropna()
        grp["data"] = df

    return data


def scale(data, scaler):
    cols = ['total_hype_level', 'current_hype_level', 'previous_hype_level',
            'posts', 'upvotes', 'comments', 'distinct_authors', "change_hype_level", "rel_change", 'price']

    for grp in data:

        df = grp["data"]
        close = df["Close"].reset_index(drop=True)
        tradeable = df["tradeable"].reset_index(drop=True)
        date = df["date"].reset_index(drop=True)

        new_df = pd.DataFrame()

        for col in cols:
            all_cols = []
            for df_col in df.columns:
                if col in df_col:
                    all_cols.append(df_col)

            temp = df[all_cols]
            temp = temp.values.T  # Transpose so we have each row as column
            temp = scaler.fit_transform(temp)
            temp = pd.DataFrame(temp.T, columns=all_cols)  # Transpose back
            new_df = new_df.merge(temp, how="right", left_index=True, right_index=True)

        new_df["Close"] = close
        new_df["tradeable"] = tradeable
        new_df["date"] = date
        grp["data"] = new_df

    return data


def pipeline(data):
    data = add_relative_change(data)
    data = add_price_col(data)
    data = add_pre_data(data, 6)
    data = scale(data, MinMaxScaler())
    return data


if __name__ == "__main__":
    with open(paths.train_path / "data_cleaned.pkl", "rb") as f:
        data = pkl.load(f)

    data = pipeline(data)

    with open(paths.train_path / "data_timeseries.pkl", "wb") as f:
        pkl.dump(data, f)
