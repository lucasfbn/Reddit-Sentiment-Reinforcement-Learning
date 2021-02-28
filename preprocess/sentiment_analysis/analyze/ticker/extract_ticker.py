import os

import pandas as pd

import paths

df = None
first = True
for filename in os.listdir(paths.ticker_folder):
    if first:
        df = pd.read_csv(paths.ticker_folder / filename, sep="\t")
        df = df[["Symbol"]]
        df["exchange"] = filename.split(".")[0]
        first = False
    else:
        next_df = pd.read_csv(paths.ticker_folder / filename, sep="\t")
        next_df = next_df[["Symbol"]]
        next_df["exchange"] = filename.split(".")[0]
        df = df.append(next_df)

df = df.sort_values(by=["Symbol"])
df.to_csv(paths.all_ticker, sep=";", index=False)
