import warnings
import os
import json
import paths
import logging
import pandas as pd
import time
from pathlib import WindowsPath
from types import SimpleNamespace

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


class Tracker:

    def __init__(self):
        self.run_id = 0
        self.arguments = {}

    def add(self, val, key=None, replace=False, only_once=False):

        if key in self.arguments and only_once:
            return
        elif key is None:
            self.arguments.update(val)
        elif key not in self.arguments or (key in self.arguments and replace):
            self.arguments[key] = [val]
        else:
            self.arguments[key].append(val)

    def _flatten(self, dic):
        new_dict = {}
        for key, value in dic.items():
            new_value = {}
            for item in value:
                new_value.update(item)
            new_dict[key] = new_value
        return new_dict

    def new(self, kind):

        # n_tracking_ids = len([_ for _ in os.listdir(paths.tracking_path / kind)
        #                       if os.path.isfile(paths.tracking_path / kind / _)])
        self.arguments = self._flatten(self.arguments)

        path = paths.tracking_path / f"{kind}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, sep=";")
            df = df.append(pd.DataFrame(self.arguments))
        else:
            df = pd.DataFrame(self.arguments)
        df.to_csv(path, sep=";", index=False)

        # with open(paths.tracking_path / kind / f"{n_tracking_ids + 1}.json", "w+") as f:
        #     json.dump(self.arguments, f)
        # log.info(f"Created tracker file: {n_tracking_ids + 1}.json")

    def add_to(self, run_id, kind):
        with open(paths.tracking_path / kind / f"{run_id}.json") as f:
            data = json.load(f)

        for key, value in self.arguments.items():
            if key in data:
                data[key].append(value)
            else:
                data[key] = value

        data = self._flatten(data)

        with open(paths.tracking_path / f"{run_id}.json", "w+") as f:
            json.dump(data, f)
        log.info(f"Added to tracker file: {run_id}.json")


tracker = Tracker()


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
            elif not is_jsonable(value):
                value = value.__qualname__

            flattened[key] = [value]

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
