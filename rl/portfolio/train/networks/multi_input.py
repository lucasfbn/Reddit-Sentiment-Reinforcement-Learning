import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):
        """
        Args:
            observation_space:
            features_dim: Number of features extracted. This corresponds to the number of unit for the last layer.
        """
        super(CustomCNN, self).__init__(observation_space, features_dim=1)

        timeseries_obs = observation_space.spaces["timeseries"]
        constants_obs = observation_space.spaces["constants"]

        self.timeseries_extractor, timeseries_out_shape = self._timeseries_extractor(timeseries_obs)
        self.constants_extractor, constants_out_shape = self._constants_extractor(constants_obs)

        self._features_dim = constants_out_shape + timeseries_out_shape

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
        timeseries_obs = th.as_tensor(observations["timeseries"][None]).float()
        constants_obs = th.as_tensor(observations["constants"][None]).float()
        return th.cat([self.timeseries_extractor(timeseries_obs),
                       self.constants_extractor(constants_obs)], dim=1)


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

    # print(obs_space.sample())

    model = CustomCNN(obs_space)
    out = model(obs_space.sample())
    print(out)
