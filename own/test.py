import numpy as np
import pickle as pkl
import paths
from collections import deque

with open(paths.data_path / "data_timeseries.pkl", "rb") as f:
    data = pkl.load(f)

for grp in data:
    if grp["ticker"] == "GAXY":
        print()

print()
