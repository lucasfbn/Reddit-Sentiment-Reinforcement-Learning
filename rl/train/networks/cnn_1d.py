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
            nn.Conv1d(in_channels=obs_shape[0], out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, n_flatten // 2), nn.ReLU(),
                                    nn.Linear(n_flatten // 2, n_flatten // 4), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


if __name__ == '__main__':
    from gym import Env, spaces
    import numpy as np

    shape = (10, 14)

    obs_space = spaces.Box(low=np.zeros(shape),
                           high=np.ones(shape),
                           dtype=np.float64)

    model = CustomCNN(obs_space)
    # model = nn.Conv1d(in_channels=obs_space.shape[0], out_channels=32, kernel_size=5, stride=1, padding=0)
    out = model(th.as_tensor(obs_space.sample()[None]).float())
    print(out)
