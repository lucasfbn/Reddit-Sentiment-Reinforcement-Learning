import datetime
import pickle as pkl

import pandas as pd
import yfinance as yf

import paths
from preprocess.stock_prices import StockPrices

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

    return df


def handle_time(df, start_hour=17, start_min=0):
    df["time"] = pd.to_datetime(df["time"], format="%d-%m-%Y %H:%M")
    df = df.sort_values(by=["time"])

    df["time"] = df["time"].dt.tz_localize("UTC")
    df["time_mesz"] = df["time"].dt.tz_convert("Europe/Berlin")
    df["time_shifted"] = df["time_mesz"] - pd.Timedelta(hours=start_hour, minutes=start_min) + pd.Timedelta(days=1)

    df["date_day"] = pd.to_datetime(df['time_shifted']).dt.to_period('D')
    return df


def grp_by(df):
    grp_by = df.groupby(["ticker_symbol"])

    grps = []

    for name, group in grp_by:
        grps.append({"ticker": name, "data": group.groupby(["date_day"]).agg("sum").reset_index()})

    return grps


def drop_short(grps, min_len):
    filtered_grps = []

    for grp in grps:
        if len(grp["data"]) >= min_len:
            filtered_grps.append(grp)

    return filtered_grps


def add_stock_prices(grps, start_offset, live):
    new_grps = []

    for i, grp in enumerate(grps):
        print(f"Processing {i}/{len(grps)}")
        sp = StockPrices(grp, start_offset=start_offset, live=live)
        new_grps.append({"ticker": grp["ticker"], "data": sp.download()})
    return new_grps


def pipeline(data, min_len=7, start_offset=30, live=False):
    df = preprocess(data)
    df = handle_time(df)
    grps = grp_by(df)
    grps = drop_short(grps, min_len=min_len)
    grps = add_stock_prices(grps, start_offset, live=live)

    return grps


if __name__ == "__main__":
    df = pd.read_csv(paths.train_path / "report.csv", sep=",")
    # df.to_csv(paths.test_path / "report_de.csv", sep=";", index=False)

    data = pipeline(df)

    with open(paths.train_path / "data_offset.pkl", "wb") as f:
        pkl.dump(data, f)
