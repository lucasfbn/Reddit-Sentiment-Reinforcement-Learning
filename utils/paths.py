import os
from pathlib import Path

"""
Used to avoid absolute paths throughout the project.
"""

base_path = Path(__file__).parent
tests_path = base_path / "tests"
storage_path = base_path / "storage"
data_path = storage_path / "data"

mlflow_dir = Path("C:/project_data/Trendstuff/mlflow")
mlflow_path = f"file:///{(mlflow_dir / 'mlruns').as_posix()}"

artifact_path = lambda artifact_uri: Path("C:" + artifact_uri.split(":")[2])

price_data_cache = storage_path / "price_data_cache.sqlite"


def create_dir(path, fn="", suffix=0):
    while os.path.exists(path / (fn + f"_{suffix}")):
        suffix += 1
    os.mkdir(path / (fn + f"_{suffix}"))
    return path / (fn + f"_{suffix}")


# sentiment analysis
ticker_dir = storage_path / "ticker"
ticker_files = ticker_dir / "files"
all_ticker = ticker_dir / "ticker.csv"
