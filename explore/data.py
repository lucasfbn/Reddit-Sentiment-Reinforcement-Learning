import pickle as pkl
import paths

with open(paths.train_path / "offset_7_all_marketsymbols" / "data_cleaned.pkl", "rb") as f:
    data = pkl.load(f)

print()
