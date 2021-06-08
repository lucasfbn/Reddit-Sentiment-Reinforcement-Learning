from prefect import task
from utils.mlflow_api import log_file


@task
def mlflow_log_file(obj, fn):
    log_file(obj, fn)
