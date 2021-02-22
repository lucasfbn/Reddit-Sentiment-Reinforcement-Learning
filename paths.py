from pathlib import Path

"""
Used to avoid absolute paths throughout the project.
"""

base_path = Path(__file__).parent
data_path = base_path / "data"
models_path = base_path / "models"
tracking_path = base_path / "tracking"

data_live_path = data_path / "live"

data_overview = data_path / "overview.csv"


def d_path(folder):
    return data_path / str(folder)
