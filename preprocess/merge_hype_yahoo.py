import pandas as pd
from preprocess import yahoo as y
import pickle as pkl
import paths

pd.options.mode.chained_assignment = None


def preprocess(df):
    df = df.drop(columns=['Unnamed: 0', 'Run Id'])
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


def add_stock_prices(grps):
    new_grps = []

    for i, grp in enumerate(grps):
        print(f"Processing {i}/{len(grps)}")

        new_grps.append({"ticker": grp["ticker"], "data": y.merge(grp["data"], grp["ticker"], start_offset=30)})
    return new_grps


df = pd.read_csv(paths.data_path / "report_de.csv", sep=";")
df = preprocess(df)
grps = grp_by(df)
grps = add_stock_prices(grps)

with open(paths.data_path / "data_offset.pkl", "wb") as f:
    pkl.dump(grps, f)
