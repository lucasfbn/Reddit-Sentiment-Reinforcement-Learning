import mlflow

import paths
from utils.mlflow_api import *

mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("pytest")


def test_log_load():
    with mlflow.start_run():
        log_file({"test": "successful"}, "test_dict.json")
        result = load_file(run_id=mlflow.active_run().info.run_id, fn="test_dict.json")
    assert result == {"test": "successful"}


def test_has_active_run():
    assert MlflowAPI().has_active_run() is False

    with mlflow.start_run():
        assert MlflowAPI().has_active_run() is True
