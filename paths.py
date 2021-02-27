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
models_path = storage_path / "models"
tracking_path = storage_path / "tracking"


def d_path(folder):
    if not os.path.exists(datasets_data_path / str(folder)):
        create_dir(datasets_data_path, str(folder), "")

    return datasets_data_path / str(folder)


def s_path(folder, live=False):
    additional = ""
    if live:
        additional = "live"
    return sentiment_data_path / additional / folder


def create_dir(path, fn, suffix):
    while os.path.exists(path / (fn + f"_{suffix}")):
        suffix += 1
    os.mkdir(path / (fn + f"_{suffix}"))
    return path / (fn + f"_{suffix}")


# sentiment analysis
sentiment_analysis_path = base_path / "preprocess" / "sentiment_analysis"
ticker_folder = sentiment_analysis_path / "analyze" / "ticker" / "files"
all_ticker = sentiment_analysis_path / "analyze" / "ticker" / "ticker.csv"
