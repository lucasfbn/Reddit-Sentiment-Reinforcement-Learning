from pathlib import Path

"""
Used to avoid absolute paths throughout the project.
"""

base_path = Path(__file__).parent
data_path = base_path / "data"
models_path = base_path / "models"

train_path = data_path / "train"
test_path = data_path / "test"
live_path = data_path / "live"

data_paths = {
    0: data_path / "train",
    1: data_path / "test",
    2: data_path / "live",
    3: data_path / "look_back_7_start_22_0_min_len_7",
    4: data_path / "look_back_7_start_22_0_min_len_9"

}
