import mlflow
import paths

from utils.mlflow_api import load_file

mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("Tests")

data = load_file(run_id="316b9d31ba41439ebc34c309f9a659ee", fn="ticker.pkl", experiment="Datasets")

import pickle as pkl

with open("temp.pkl", "wb") as f:
    pkl.dump(data[:10], f)

print()
