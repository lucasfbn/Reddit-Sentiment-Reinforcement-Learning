import paths
from utils import Config
import copy

general = Config(**dict(
    data_path=paths.datasets_data_path / "_1" / "timeseries.pkl",
    kind="CNN",
    evaluate=False,
    model_path=paths.models_path / "14-09 13_03-21"
))

agent = Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.975,
    epsilon_min=0.10,
    randomness=True,
    memory_len=1000,
))

model = Config(**dict(
    n_episodes=3,
    batch_size=32
))

config = Config(**dict(general=general, agent=agent, model=model))

configs = [
    [Config(**dict(
        data_path=paths.datasets_data_path / "_13" / "timeseries.pkl",
        kind="CNN",
        evaluate=False,
        model_path=None
    )), Config(**dict(
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.15,
        randomness=True,
        memory_len=1000,
    )), Config(**dict(
        n_episodes=3,
        batch_size=32
    ))],
    [Config(**dict(
        data_path=paths.datasets_data_path / "_13" / "timeseries.pkl",
        kind="CNN",
        evaluate=False,
        model_path=None
    )), Config(**dict(
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.15,
        randomness=False,
        memory_len=1000,
    )), Config(**dict(
        n_episodes=3,
        batch_size=32
    ))],
    [Config(**dict(
        data_path=paths.datasets_data_path / "_2" / "timeseries.pkl",
        kind="CNN",
        evaluate=False,
        model_path=None
    )), Config(**dict(
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.15,
        randomness=True,
        memory_len=1000,
    )), Config(**dict(
        n_episodes=3,
        batch_size=32
    ))],
    [Config(**dict(
        data_path=paths.datasets_data_path / "_2" / "timeseries.pkl",
        kind="CNN",
        evaluate=False,
        model_path=None
    )), Config(**dict(
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.15,
        randomness=False,
        memory_len=1000,
    )), Config(**dict(
        n_episodes=3,
        batch_size=32
    ))],
]
