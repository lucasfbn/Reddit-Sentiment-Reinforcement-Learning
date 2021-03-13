import json
import logging
import os
import time
from datetime import datetime
from pathlib import WindowsPath
from types import SimpleNamespace

import pandas as pd

import paths

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


def save_config(configs, kind):
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

            flattened[key] = [value]

    flattened["time"] = datetime.now()

    path = paths.tracking_path / f"{kind}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, sep=";")
        df = df.append(pd.DataFrame(flattened))
    else:
        df = pd.DataFrame(flattened)

    while True:
        try:
            df.to_csv(path, sep=";", index=False)
            break
        except PermissionError:
            print(f"Close {kind}.csv you retard.")
            time.sleep(3)


class Config(SimpleNamespace):
    pass


if __name__ == '__main__':
    c = Config(**dict(test=12))
    save_config([c], "test")
