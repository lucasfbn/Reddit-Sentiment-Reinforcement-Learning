import json
import pickle as pkl
import tempfile
from pathlib import Path

import mlflow
import pandas as pd


def _get_kind(fn):
    kind = fn.split(".")[1]
    assert kind in ["pkl", "json", "csv", "png"]
    return kind


def _artifact_path(artifact_uri):
    return Path("C:/" + artifact_uri.split(":")[2])


def log_file(file, fn):
    """
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Testing")  #
    mlflow.start_run()

    mlflow_log_file({"test": 1}, "test.json")

    mlflow.end_run()
    """

    kind = _get_kind(fn)

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


def load_file(run_id, fn, experiment=None):
    kind = _get_kind(fn)

    if experiment is not None:
        active_run = mlflow.active_run()
        active_experiment = mlflow.get_experiment(active_run.info.experiment_id).name
        mlflow.set_experiment(experiment)

    from_run = mlflow.get_run(run_id)
    from_artifact_path = _artifact_path(from_run.info.artifact_uri)

    if kind == "pkl":
        with open(from_artifact_path / fn, "rb") as f:
            file = pkl.load(f)
    elif kind == "json":
        with open(from_artifact_path / fn) as f:
            file = json.load(f)
    elif kind == "csv":
        file = pd.read_csv(from_artifact_path / fn, sep=";")
    else:
        raise NotImplementedError

    if experiment is not None:
        mlflow.set_experiment(active_experiment)

    return file
