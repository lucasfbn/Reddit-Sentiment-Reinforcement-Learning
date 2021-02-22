from pathlib import Path

"""
Used to avoid absolute paths throughout the project.
"""

base_path = Path(__file__).parent

storage_path = base_path / "storage"
data_path = storage_path / "data"
models_path = storage_path / "models"
tracking_path = storage_path / "tracking"


def d_path(folder):
    return data_path / str(folder)


# sentiment analysis
sentiment_analysis_path = base_path / "preprocess" / "sentiment_analysis"
ticker_folder = sentiment_analysis_path / "analyze" / "ticker" / "files"
all_ticker = sentiment_analysis_path / "analyze" / "ticker" / "ticker.csv"
