import pickle as pkl
import paths
import pandas as pd

# p = r"C:\Users\lucas\OneDrive\Backup\Projects\Trendstuff\storage\data\sentiment\01-11-21 - 30-11-21_0_MANUAL\gc_dump.csv"
# df = pd.read_csv(p, sep=";")
#
with open(paths.datasets_data_path / "_1" / "timeseries.pkl", "rb") as f:
    data1 = pkl.load(f)


l = 0
for d in data1:
    l += len(d["data"])

print(l)