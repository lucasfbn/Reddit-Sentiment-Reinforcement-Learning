from utils.mlflow_api import load_file
import pandas as pd
import pickle as pkl

# ticker = load_file(fn="ticker.pkl", run_id="32b8ae74f0c143349cb7777fd2c1dcb5", experiment="Live")

with open("temp.pkl", "rb") as f:
    ticker = pkl.load(f)

print()

df = pd.DataFrame()

for seq in ticker.sequences:
    seq.arr.loc[len(seq.arr) - 2, "seq_price"] = seq.price
    df = df.append(seq.arr)
    df = df.append(pd.Series(), ignore_index=True)

print()

df.to_csv("temp.csv", sep=";")
