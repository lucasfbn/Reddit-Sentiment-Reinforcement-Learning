import os
from pathlib import Path

"""
Used to avoid absolute paths throughout the project.
"""

base_path = Path(__file__).parent.parent
tests_path = base_path / "tests"
storage_path = base_path / "storage"
data_path = storage_path / "data"

mlflow_dir = Path("C:/project_data/Trendstuff/mlflow")

price_data_cache = storage_path / "price_data_cache.sqlite"

# sentiment analysis
ticker_dir = storage_path / "ticker"
ticker_files = ticker_dir / "files"
all_ticker = ticker_dir / "ticker.csv"
