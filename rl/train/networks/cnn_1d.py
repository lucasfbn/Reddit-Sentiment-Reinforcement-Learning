import gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        Args:
            observation_space:
            features_dim: Number of features extracted. This corresponds to the number of unit for the last layer.
        """
        super(CustomCNN, self).__init__(observation_space, features_dim)

        obs_shape = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=obs_shape[0], out_channels=32, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
