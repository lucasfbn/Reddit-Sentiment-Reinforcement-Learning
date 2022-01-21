import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Network(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim):
        """
        Args:
            observation_space:
        """
        super(Network, self).__init__(observation_space, features_dim=features_dim)

        timeseries_obs = observation_space.spaces["timeseries"]
        constants_obs = observation_space.spaces["constants"]

        self.timeseries_extractor, timeseries_out_shape = self._timeseries_extractor(timeseries_obs)
        self.constants_extractor, constants_out_shape = self._constants_extractor(constants_obs)

        self.out_layer = nn.Sequential(nn.Linear(constants_out_shape + timeseries_out_shape, features_dim),
                                       nn.ReLU())

    def _constants_extractor(self, subspace):
        out_neurons = 16
        network = nn.Sequential(nn.Linear(subspace.shape[0], out_neurons), nn.ReLU())

        return network, out_neurons

    def _timeseries_extractor(self, subspace):
        network = nn.Sequential(
            nn.Conv1d(in_channels=subspace.shape[0], out_channels=64, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            out_shape = network(
                th.as_tensor(subspace.sample()[None]).float()
            ).shape[1]

        return network, out_shape

    def forward(self, observations: th.Tensor) -> th.Tensor:
        timeseries_obs = th.as_tensor(observations["timeseries"]).float()
        constants_obs = th.as_tensor(observations["constants"]).float()

        cat_ = th.cat([self.timeseries_extractor(timeseries_obs),
                       self.constants_extractor(constants_obs)], dim=1)
        return self.out_layer(cat_)


if __name__ == '__main__':
    from gym import spaces
    import numpy as np

    timeseries_shape = (10, 14)
    constants_shape = (3)

    obs_space = spaces.Dict(
        {"timeseries": spaces.Box(low=np.zeros(timeseries_shape),
                                  high=np.ones(timeseries_shape),
                                  dtype=np.float64),
         "constants": spaces.Box(low=np.zeros(constants_shape),
                                 high=np.ones(constants_shape),
                                 dtype=np.float64)}
    )

    obs = obs_space.sample()
    # Add batch dimension
    obs["timeseries"] = np.expand_dims(obs["timeseries"], axis=0)
    obs["constants"] = np.expand_dims(obs["constants"], axis=0)

    model = Network(obs_space, 126)
    out = model.forward(obs)
    print(out)
