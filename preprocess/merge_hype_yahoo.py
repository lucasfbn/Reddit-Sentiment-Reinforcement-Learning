import datetime
import pickle as pkl

import pandas as pd
import yfinance as yf

import paths

pd.options.mode.chained_assignment = None


def preprocess(df):
    df = df.drop(columns=['Unnamed: 0', 'Run Id'], errors="ignore")
    df = df.rename({"Run Time (UTC)": "time"}, axis=1)

    new_cols = []
    for col in df.columns:
        col = col.lower()
        col = col.replace(" #", "")
        col = col.strip()
        col = col.replace(" ", "_")
        new_cols.append(col)

    df.columns = new_cols
    df["time"] = pd.to_datetime(df["time"], format="%d-%m-%Y %H:%M")
    df = df.sort_values(by=["time"])
    df["date_day"] = pd.to_datetime(df['time']).dt.to_period('D')
    return df


def grp_by(df):
    grp_by = df.groupby(["ticker_symbol"])

    grps = []

    for name, group in grp_by:
        grps.append({"ticker": name, "data": group.groupby(["date_day"]).agg("sum").reset_index()})

    return grps


def drop_short(grps, min_len=7):
    filtered_grps = []

    for grp in grps:
        if len(grp["data"]) >= min_len:
            filtered_grps.append(grp)

    return filtered_grps


def add_stock_prices(grps):

    def merge(df, symbol, start_offset):
        start = str(df["date_day"].min() - datetime.timedelta(days=start_offset))
        end = str(df["date_day"].max())

        historical_data = yf.download(symbol, start=start, end=end)
        historical_data["date_day"] = pd.to_datetime(historical_data.index).to_period('D')

        df = historical_data.merge(df, on="date_day", how="outer")
        return df

    new_grps = []

    for i, grp in enumerate(grps):
        print(f"Processing {i}/{len(grps)}")

        new_grps.append({"ticker": grp["ticker"], "data": merge(grp["data"], grp["ticker"], start_offset=30)})
    return new_grps


df = pd.read_csv(paths.test_path / "report.csv", sep=",")
# df.to_csv(paths.test_path / "report_de.csv", sep=";", index=False)

df = preprocess(df)
grps = grp_by(df)
grps = drop_short(grps)
grps = add_stock_prices(grps)

with open(paths.test_path / "data_offset.pkl", "wb") as f:
    pkl.dump(grps, f)
