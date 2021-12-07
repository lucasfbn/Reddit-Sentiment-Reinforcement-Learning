import gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Networks:

    def __init__(self, in_channels):
        self.in_channels = in_channels

    @property
    def networks(self):
        return [
            (nn.Sequential(
                nn.Conv1d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            ), 96),
            (nn.Sequential(
                nn.Conv1d(in_channels=self.in_channels, out_channels=64, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            ), 64),
            (nn.Sequential(
                nn.Conv1d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=7, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            ), 16)
        ]


class TuneableNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, cnn: nn.Sequential, features_dim: int = 96):
        """
        Args:
            observation_space:
            features_dim: Number of features extracted. This corresponds to the number of unit for the last layer.
        """
        super(TuneableNetwork, self).__init__(observation_space, features_dim)

        self.cnn = cnn

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, n_flatten // 2), nn.ReLU(),
                                    nn.Linear(n_flatten // 2, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


if __name__ == '__main__':

    from gym import Env, spaces
    import numpy as np

    shape = (10, 14)

    obs_space = spaces.Box(low=np.zeros(shape),
                           high=np.ones(shape),
                           dtype=np.float64)

    networks = Networks(in_channels=obs_space.shape[0])

    for network in networks.networks:
        model = TuneableNetwork(obs_space, network)
        out = model(th.as_tensor(obs_space.sample()[None]).float())
        print(out.shape)
        # print(out)
