from abc import ABC, abstractmethod

import numpy as np
from gym import Env, spaces

from preprocessing.sequences import Sequence
from rl.train.envs.state_handler.state_handler import StateHandlerCNN, StateHandlerNN
from rl.train.envs.sub_envs.trading import SimpleTradingEnvTraining
from rl.train.envs.utils.data_iterator import DataIterator
from rl.train.envs.utils.reward_counter import RewardCounter


class BaseEnv(Env, ABC):

    def __init__(self, ticker):
        super().__init__()

        self.data_iter = DataIterator(ticker)
        self.reward_counter = RewardCounter()
        self.trading_env = SimpleTradingEnvTraining("init")

        self.action_space = spaces.Discrete(3, )
        shape = self._get_initial_observation_state_shape()
        self.observation_space = spaces.Box(low=np.zeros(shape),
                                            high=np.ones(shape),
                                            dtype=np.float64)

    @abstractmethod
    def forward_state(self, sequence: Sequence):
        pass

    def _get_first_sequence(self):
        return self.data_iter.ticker[0].sequences[0]

    def _get_initial_observation_state_shape(self):
        return self.forward_state(self._get_first_sequence()).shape

    def step(self, actions):
        price = self.data_iter.curr_sequence.price

        # Hold
        if actions == 0:
            reward = self.trading_env.hold(price)

        # Buy
        elif actions == 1:
            reward = self.trading_env.buy(price)

        # Sell
        elif actions == 2:
            reward = self.trading_env.sell(price)

        else:
            raise ValueError("Invalid action.")

        self.reward_counter.add_reward(reward)

        next_sequence = self.data_iter.next_sequence()
        next_state = self.forward_state(next_sequence)

        return next_state, reward, self.data_iter.is_episode_end(), {}

    def close(self):
        pass

    def reset(self):
        self.data_iter.episode_end = False
        self.data_iter.episode_count += 1

        self.data_iter.next_ticker()
        next_sequence = self.data_iter.next_sequence()
        state = self.forward_state(next_sequence)
        self.trading_env = SimpleTradingEnvTraining(ticker_name=self.data_iter.curr_ticker.name)
        return state

    def log(self):
        self.reward_counter.log(step=self.data_iter.episode_count)
        self.reward_counter = RewardCounter()


class EnvNN(BaseEnv):
    state_handler = StateHandlerNN()

    def forward_state(self, sequence: Sequence):
        self.state_handler.forward(sequence)


class EnvNNExtended(EnvNN):
    def forward_state(self, sequence: Sequence):
        self.state_handler.forward_extend(sequence, self.trading_env.inventory_state())


class EnvCNN(BaseEnv):
    state_handler = StateHandlerCNN()

    def forward_state(self, sequence: Sequence):
        self.state_handler.forward(sequence)


class EnvCNNExtended(EnvCNN):
    def forward_state(self, sequence: Sequence):
        self.state_handler.forward_extend(sequence, self.trading_env.inventory_state())
