import os
from pathlib import Path

"""
Used to avoid absolute paths throughout the project.
"""

base_path = Path(__file__).parent

storage_path = base_path / "storage"
data_path = storage_path / "data"
mlflow_path = f"file:///{(storage_path / 'mlflow' / 'mlruns').as_posix()}"


def create_dir(path, fn="", suffix=0):
    while os.path.exists(path / (fn + f"_{suffix}")):
        suffix += 1
    os.mkdir(path / (fn + f"_{suffix}"))
    return path / (fn + f"_{suffix}")


# sentiment analysis
sentiment_analysis_path = base_path / "sentiment_analysis"
ticker_folder = sentiment_analysis_path / "analyze" / "ticker" / "files"
all_ticker = sentiment_analysis_path / "analyze" / "ticker" / "ticker.csv"
