from pathlib import Path

"""
Used to avoid absolute paths throughout the project.
"""

base_path = Path(__file__).parent.parent
tests_path = base_path / "tests"
data_path = base_path / "data"

price_data_cache = data_path / "price_data_cache.sqlite"

# sentiment analysis
ticker_dir = data_path / "ticker"
ticker_files = ticker_dir / "files"
all_ticker = ticker_dir / "ticker.csv"
