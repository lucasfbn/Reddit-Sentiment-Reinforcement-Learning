import mlflow

import paths
from utils.mlflow_api import load_file

mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("Tests")

data = load_file(run_id="247244f584294534a0d758ba65ef1749", fn="ticker.pkl", experiment="Datasets")

print()
