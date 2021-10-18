import logging

from utils.mlflow_api import MlflowAPI

format_ = "%(levelname)s %(asctime)s - %(message)s"


def setup_logger(level):
    if not MlflowAPI().has_active_run():
        return

    # Create new logger
    formatter = logging.Formatter(format_)

    log = logging.getLogger()
    file_handler = logging.FileHandler(MlflowAPI().get_artifact_path() / "debug.log", "w+")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Remove old logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    log.setLevel(level)

    return log


def base_logger(level):
    logging.basicConfig(format=format_)
    log = logging.getLogger("root")
    log.setLevel(level)
    return log
