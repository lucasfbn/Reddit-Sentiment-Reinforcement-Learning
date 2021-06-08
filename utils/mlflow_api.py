import json
import pickle as pkl
import tempfile
from pathlib import Path

import mlflow


def log_file(file, fn):
    """
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Testing")  #
    mlflow.start_run()

    mlflow_log_file({"test": 1}, "test.json")

    mlflow.end_run()
    """

    kind = fn.split(".")[1]
    assert kind in ["pkl", "json", "csv"]

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname_path = Path(tmpdirname)

        if kind == "pkl":
            with open(tmpdirname_path / fn, "wb") as f:
                pkl.dump(file, f)
        elif kind == "json":
            with open(tmpdirname_path / fn, "w+") as f:
                json.dump(file, f)
        elif kind == "csv":
            file.to_csv(tmpdirname_path / fn, sep=";", index=False)

        mlflow.log_artifact((tmpdirname_path / fn).as_posix())