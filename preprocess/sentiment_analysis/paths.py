from pathlib import Path

"""
Used to avoid absolute paths throughout the project.
"""

base_path = Path(__file__).parent

from pathlib import Path
import os

p = Path(os.getcwd())
print([a for a in os.listdir() if os.path.isdir(a)])