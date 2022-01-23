import wandb
import json
import pickle as pkl
import pandas as pd
import tempfile
from pathlib import Path
import os

NAME = "lucasfbn"
PROJECT = "Trendstuff"


class Handler:
    def load(self, fp):
        raise NotImplementedError

    def save(self, data, fp):
        raise NotImplementedError


class JSONHandler(Handler):
    def load(self, fp):
        with open(fp) as f:
            file = json.load(f)
        return file

    def save(self, data, fp):
        with open(fp, "w+") as f:
            json.dump(data, f)


class PklHandler(Handler):
    def load(self, fp):
        with open(fp, "rb") as f:
            file = pkl.load(f)
        return file

    def save(self, data, fp):
        with open(fp, "wb") as f:
            pkl.dump(data, f)


class CSVHandler(Handler):
    def load(self, fp):
        return pd.read_csv(fp)

    def save(self, data, fp):
        data.to_csv(fp, index=False)


HANDLER_MAPPING = {"json": JSONHandler,
                   "pkl": PklHandler,
                   "csv": CSVHandler}


def _get_file_kind(fn):
    return fn.split(".")[1]


def _check_file_kind(kind):
    if kind not in HANDLER_MAPPING.keys():
        raise NotImplementedError(f"File handler for kind '{kind}' not implemented.")


def _get_handler(fn):
    kind = _get_file_kind(fn)
    _check_file_kind(kind)
    return HANDLER_MAPPING[kind]()


def log_file(file, fn, run):
    handler = _get_handler(fn)
    path = Path(Path(run.dir) / fn).as_posix()
    handler.save(data=file, fp=path)


def log_artefact(file, fn, type):
    handler = _get_handler(fn)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname_path = Path(tmpdirname)
        fp = tmpdirname_path / fn

        handler.save(data=file, fp=fp)

        wandb.log_artifact((tmpdirname_path / fn).as_posix(), name=fn, type=type)


def load_file(run_id, fn, run):
    path = Path(wandb.restore(name=fn, run_path=f'{run.entity}/{run.project}/{run_id}').name)
    handler = _get_handler(fn)
    return handler.load(path)


def load_artefact(run, fn, version, type):
    art = run.use_artifact(f'{run.entity}/{run.project}/{fn}:v{version}', type=type)
    path = Path(art.download(root=os.getenv("WANDB_DIR") + "artefacts")) / fn

    handler = _get_handler(fn)
    return handler.load(path)


def get_histories(ids, col):
    api = wandb.Api()

    r = []

    for id_ in ids:
        run = api.run(f"{NAME}/{PROJECT}/{id_}")
        history = run.history(samples=9999999)
        df = history[col]
        df = df.dropna()
        df = df.reset_index(drop=True)
        r.append((id_, df))

    return r


def log_to_summary(run, d: dict):
    summary = run.summary

    for key, value in d.items():
        summary[key] = value
        