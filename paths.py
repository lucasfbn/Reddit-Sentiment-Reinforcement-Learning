from pathlib import Path

"""
Used to avoid absolute paths throughout the project.
"""

base_path = Path(__file__).parent
data_path = base_path / "data"
models_path = base_path / "models"