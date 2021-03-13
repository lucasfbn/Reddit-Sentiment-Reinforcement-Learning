import paths
from utils import Config
from learning.main import main

general = Config(**dict(
    data_path=paths.datasets_data_path / "_13" / "timeseries.pkl",
    kind="CNN",
    eval=True,
    model_path=paths.models_path / "10-52 12_03-21"
))

agent = Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.999,
    epsilon_min=0.15,
    randomness=True,
    memory_len=1000,
))

# main(config)
