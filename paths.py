from pathlib import Path

"""
Used to avoid absolute paths throughout the project.
"""

base_path = Path(__file__).parent
data_path = base_path / "data"
models_path = base_path / "models"

data_live_path = data_path / "live"

data_overview = data_path / "overview.csv"
data_paths = {
    0: data_path / "train",
    1: data_path / "test",
    2: None,
    3: data_path / "3",
    4: data_path / "4",
    5: data_path / "5",
    6: data_live_path / "17_02_2021",
    10: data_path / "10"
}
