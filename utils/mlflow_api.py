import json
import pickle as pkl
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

import paths


def init_mlflow(experiment):
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment(experiment)


class MlflowAPI:
    valid_files_kinds = ["pkl", "json", "csv", "png"]

    def __init__(self, run_id=None, experiment=None):
        self.experiment = experiment
        self.run_id = run_id

        self._active_run = self._get_active_run()
        self._active_exp = self._get_active_experiment()

        self._relevant_run_id = None
        self._relevant_run_obj = None
        self._relevant_exp = None

        self._get_relevant_run()
        self._get_relevant_exp()

    def has_active_run(self):
        if self._get_active_run() is None:
            return False
        return True

    def _get_active_run(self):
        return mlflow.active_run()

    def _get_active_experiment(self):
        if self.has_active_run():
            return mlflow.get_experiment(self._active_run.info.experiment_id).name
        return None

    def _set_experiment(self, experiment):
        if experiment is not None:
            mlflow.set_experiment(experiment)

    def _get_relevant_run(self):
        self._relevant_run_id = self.run_id if self.run_id is not None else self._active_run
        self._relevant_run_obj = mlflow.get_run(self.run_id) if self.run_id is not None else mlflow.active_run()

    def _get_relevant_exp(self):
        self._relevant_exp = self.experiment if self.experiment is not None else self._active_exp

    def get_artifact_path(self):
        r = Path("C:/" + self._relevant_run_obj.info.artifact_uri.split(":")[2])
        self._exit()
        return r

    def _get_file_kind(self, fn):
        kind = fn.split(".")[1]
        assert kind in self.valid_files_kinds
        return kind

    def _exit(self):
        self._set_experiment(self._active_exp)

    def log_file(self, file, fn):
        kind = self._get_file_kind(fn)

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

        self._exit()

    def load_file(self, fn):
        self._set_experiment(self._relevant_exp)
        artifact_path = self.get_artifact_path()

        kind = self._get_file_kind(fn)

        if kind == "pkl":
            with open(artifact_path / fn, "rb") as f:
                file = pkl.load(f)
        elif kind == "json":
            with open(artifact_path / fn) as f:
                file = json.load(f)
        elif kind == "csv":
            file = pd.read_csv(artifact_path / fn, sep=";")
        else:
            raise NotImplementedError

        self._exit()

        return file


def log_file(file, fn):
    MlflowAPI().log_file(file, fn)


def load_file(fn, run_id=None, experiment=None):
    return MlflowAPI(run_id=run_id, experiment=experiment).load_file(fn)
