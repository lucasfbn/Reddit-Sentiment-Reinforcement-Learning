import json
import logging
from datetime import datetime
from pathlib import WindowsPath
from types import SimpleNamespace

import mlflow
import pandas as pd

import paths
from mlflow_api import log_file

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(asctime)s - %(message)s")
log = logging.getLogger()


def drop_stats(func):
    def wrapper(*args, **kwargs):
        old_len = len(args[0].df)
        func(*args, **kwargs)
        log.info(f"{func.__name__} dropped {old_len - len(args[0].df)} items.")

    return wrapper


def dt_to_timestamp(time):
    if time is None: return None
    return int(time.timestamp())


def save_config(configs):
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    flattened = {}

    for config in configs:
        config = vars(config)
        for key, value in config.items():

            if isinstance(value, WindowsPath):
                value = str(value.name)
            elif isinstance(value, datetime):
                value = str(value)
            elif not is_jsonable(value):
                value = value.__qualname__

            flattened[key] = value

    mlflow.log_params(flattened)


class Config(SimpleNamespace):
    pass


if __name__ == '__main__':

    df = pd.DataFrame({"Hallo": [1,2,3], "Tsch": [4,5,6]})

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Testing")  #
    mlflow.start_run()

    log_file(df, "test.csv")

    mlflow.end_run()