import mlflow
import utils.paths

from mlflow_utils import init_mlflow, load_file

init_mlflow(utils.paths.mlflow_dir, "Tests")

data = load_file(run_id="c3925e0cbdcd4620b5cb909e1a629419", fn="ticker.pkl", experiment="Datasets")

import pickle as pkl

with open("temp.pkl", "wb") as f:
    pkl.dump(data[:10], f)

print()
