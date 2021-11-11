import json
import logging
from datetime import datetime
from pathlib import WindowsPath
from types import SimpleNamespace

import mlflow
import pandas as pd

import paths
from utils.mlflow_api import log_file

# Backwards compatibility, use utils/logger.
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(asctime)s - %(message)s")
log = logging.getLogger()


def update_check_key(input_dict, update_dict):
    for key in update_dict.keys():
        if key not in input_dict.keys():
            raise ValueError(f"Invalid key: {key}")
    input_dict.update(update_dict)
    return input_dict


def drop_stats(func):
    def wrapper(*args, **kwargs):
        old_len = len(args[0].df)
        func(*args, **kwargs)
        log.info(f"{func.__name__} dropped {old_len - len(args[0].df)} items.")

    return wrapper


def dt_to_timestamp(time):
    if time is None: return None
    return int(time.timestamp())
