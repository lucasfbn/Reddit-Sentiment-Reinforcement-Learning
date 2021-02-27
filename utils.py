import warnings
import os
import json
import paths
import logging

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

    def add(self, val, key=None):
        if key is None:
            self.arguments.update(val)
        elif key not in self.arguments:
            self.arguments[key] = [val]
        else:
            self.arguments[key].append(val)

    def new(self):
        n_tracking_ids = len([_ for _ in os.listdir(paths.tracking_path) if os.path.isfile(paths.tracking_path / _)])
        with open(paths.tracking_path / f"{n_tracking_ids + 1}.json", "w+") as f:
            json.dump(self.arguments, f)
        log.info(f"Created tracker file: {n_tracking_ids + 1}.json")

    def add_to(self, run_id):
        with open(paths.tracking_path / f"{run_id}.json") as f:
            data = json.load(f)

        for key, value in self.arguments.items():
            if key in data:
                data[key].extend(value)
            else:
                data[key] = value

        with open(paths.tracking_path / f"{run_id}.json", "w+") as f:
            json.dump(data, f)
        log.info(f"Added to tracker file: {run_id}.json")


if __name__ == '__main__':
    tracker = Tracker()
    tracker.add({"test5": 1}, "hallo1")
    # tracker.new()
    tracker.add_to(5)
