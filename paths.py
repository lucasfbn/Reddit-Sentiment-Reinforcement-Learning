import os
from pathlib import Path

"""
Used to avoid absolute paths throughout the project.
"""

base_path = Path(__file__).parent

storage_path = base_path / "storage"
data_path = storage_path / "data"
datasets_data_path = data_path / "datasets"
sentiment_data_path = data_path / "sentiment"
eval_data_path = storage_path / "eval"
models_path = storage_path / "models"
models_temp_path = models_path / "temp"
tracking_path = storage_path / "tracking"
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
