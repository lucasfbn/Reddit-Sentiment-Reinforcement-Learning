import paths
import pandas as pd
import pickle as pkl
import numpy as np
import datetime
import preprocess.yahoo as yh

with open(paths.data_path / "data.pkl", "rb") as f:
    data = pkl.load(f)

for grp in data:
    grp["data"] = grp["data"].drop(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])

t = data[0]["data"]

y = yh.merge(t, "AABB", 30)

print()
