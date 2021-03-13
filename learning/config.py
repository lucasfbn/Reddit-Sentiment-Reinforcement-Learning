import paths
from utils import Config
from learning.main import main

# general = Config(**dict(
#     data_path=paths.eval_data_path / "21-50 04_03-21.pkl",
#     kind="CNN",
#     eval=False,
#     model_path=None
# ))
#
# agent = Config(**dict(
#     gamma=0.95,
#     epsilon=1.0,
#     epsilon_decay=0.999,
#     epsilon_min=0.15,
#     randomness=True,
#     memory_len=1000,
# ))

configs = [[Config(**dict(
    data_path=paths.eval_data_path / "21-50 04_03-21.pkl",
    kind="CNN",
    eval=False,
    model_path=None
)), Config(**dict(
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.999,
    epsilon_min=0.15,
    randomness=True,
    memory_len=1000,
))],
           [Config(**dict(
               data_path=paths.eval_data_path / "21-50 04_03-21.pkl",
               kind="CNN",
               eval=False,
               model_path=None
           )), Config(**dict(
               gamma=0.95,
               epsilon=1.0,
               epsilon_decay=0.999,
               epsilon_min=0.15,
               randomness=False,
               memory_len=1000,
           ))]

           ]

new_configs = []
for c in configs:
    new_configs.append(c * 3)

# configs = [(general, agent)]

for nc in new_configs:
    config = Config(**dict(
        general=nc[0],
        agent=nc[1]
    ))
    main(config)
