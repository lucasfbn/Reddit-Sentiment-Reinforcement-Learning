import paths
from utils import Config
import copy

general = Config(**dict(
    data_path=paths.datasets_data_path / "_0" / "timeseries.pkl",
    kind="NN",
    evaluate=False,
    model_path=None
))

agent = Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    randomness=True,
    memory_len=1000,
))

model = Config(**dict(
    name="nn",
    n_episodes=100,
    batch_size=32
))

config = Config(**dict(general=general, agent=agent, model=model))

c1 = [Config(**dict(
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
    name="cnn1",
    n_episodes=3,
    batch_size=32
))]

c2 = [Config(**dict(
    data_path=paths.datasets_data_path / "_13" / "timeseries.pkl",
    kind="CNN",
    evaluate=False,
    model_path=None
)), Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.975,
    epsilon_min=0.15,
    randomness=True,
    memory_len=1000,
)), Config(**dict(
    name="cnn2",
    n_episodes=3,
    batch_size=32
))]

c3 = [Config(**dict(
    data_path=paths.datasets_data_path / "_13" / "timeseries.pkl",
    kind="CNN",
    evaluate=False,
    model_path=None
)), Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.975,
    epsilon_min=0.15,
    randomness=True,
    memory_len=1000,
)), Config(**dict(
    name="cnn3",
    n_episodes=3,
    batch_size=32
))]

c4 = [Config(**dict(
    data_path=paths.datasets_data_path / "_13" / "timeseries.pkl",
    kind="CNN",
    evaluate=False,
    model_path=None
)), Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.975,
    epsilon_min=0.15,
    randomness=True,
    memory_len=1000,
)), Config(**dict(
    name="cnn4",
    n_episodes=3,
    batch_size=32
))]

c5 = [Config(**dict(
    data_path=paths.datasets_data_path / "_13" / "timeseries.pkl",
    kind="CNN",
    evaluate=False,
    model_path=None
)), Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.975,
    epsilon_min=0.15,
    randomness=True,
    memory_len=1000,
)), Config(**dict(
    name="cnn5",
    n_episodes=3,
    batch_size=32
))]

c6 = [Config(**dict(
    data_path=paths.datasets_data_path / "_13" / "timeseries.pkl",
    kind="CNN",
    evaluate=False,
    model_path=None
)), Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.975,
    epsilon_min=0.15,
    randomness=True,
    memory_len=1000,
)), Config(**dict(
    name="cnn6",
    n_episodes=3,
    batch_size=32
))]

c_1 = [Config(**dict(
    data_path=paths.datasets_data_path / "_1" / "timeseries.pkl",
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
    name="cnn1",
    n_episodes=3,
    batch_size=32
))]

c_2 = [Config(**dict(
    data_path=paths.datasets_data_path / "_1" / "timeseries.pkl",
    kind="CNN",
    evaluate=False,
    model_path=None
)), Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.975,
    epsilon_min=0.15,
    randomness=True,
    memory_len=1000,
)), Config(**dict(
    name="cnn2",
    n_episodes=3,
    batch_size=32
))]

c_3 = [Config(**dict(
    data_path=paths.datasets_data_path / "_1" / "timeseries.pkl",
    kind="CNN",
    evaluate=False,
    model_path=None
)), Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.975,
    epsilon_min=0.15,
    randomness=True,
    memory_len=1000,
)), Config(**dict(
    name="cnn3",
    n_episodes=3,
    batch_size=32
))]

c_4 = [Config(**dict(
    data_path=paths.datasets_data_path / "_1" / "timeseries.pkl",
    kind="CNN",
    evaluate=False,
    model_path=None
)), Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.975,
    epsilon_min=0.15,
    randomness=True,
    memory_len=1000,
)), Config(**dict(
    name="cnn4",
    n_episodes=3,
    batch_size=32
))]

c_5 = [Config(**dict(
    data_path=paths.datasets_data_path / "_1" / "timeseries.pkl",
    kind="CNN",
    evaluate=False,
    model_path=None
)), Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.975,
    epsilon_min=0.15,
    randomness=True,
    memory_len=1000,
)), Config(**dict(
    name="cnn5",
    n_episodes=3,
    batch_size=32
))]

c_6 = [Config(**dict(
    data_path=paths.datasets_data_path / "_1" / "timeseries.pkl",
    kind="CNN",
    evaluate=False,
    model_path=None
)), Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.975,
    epsilon_min=0.15,
    randomness=True,
    memory_len=1000,
)), Config(**dict(
    name="cnn6",
    n_episodes=3,
    batch_size=32
))]

configs = [
    copy.deepcopy(c2),  # Mehr Filter
    copy.deepcopy(c3), copy.deepcopy(c3),  # Weniger Filter
    copy.deepcopy(c4), copy.deepcopy(c4),  # Größerer Kernel
    copy.deepcopy(c5), copy.deepcopy(c5),  # Mehr Layer
    copy.deepcopy(c6), copy.deepcopy(c6),  # Mehr Layer, größerer kernel

    # Neues dataset
    copy.deepcopy(c_1), copy.deepcopy(c_1), copy.deepcopy(c_1),
    copy.deepcopy(c_2), copy.deepcopy(c_2),  # Mehr Filter
    copy.deepcopy(c_3), copy.deepcopy(c_3),  # Weniger Filter
    copy.deepcopy(c_4), copy.deepcopy(c_4),  # Größerer Kernel
    copy.deepcopy(c_5), copy.deepcopy(c_5),  # Mehr Layer
    copy.deepcopy(c_6), copy.deepcopy(c_6),  # Mehr Layer, größerer kernel
]
