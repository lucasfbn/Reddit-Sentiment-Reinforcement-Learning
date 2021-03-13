import paths
from utils import Config
from learning.main import main

config = Config(**dict(
    data_path=paths.datasets_data_path / "_13" / "timeseries.pkl",
    kind="CNN",
    eval=True,
    model_path=paths.models_path / "10-52 12_03-21"
))

main(config)
