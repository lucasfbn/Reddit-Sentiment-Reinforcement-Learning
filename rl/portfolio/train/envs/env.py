from abc import ABC

import numpy as np
from gym import Env, spaces

from rl.portfolio.train.envs.sub_envs.trading import TradingSimulator
from rl.portfolio.train.envs.utils.data_iterator import DataIterator
from rl.stocks.train.envs.state_handler.state_handler import StateHandlerCNN, StateHandlerNN


class BaseEnv(Env, ABC):

    def __init__(self, buys):
        super().__init__()

        self._buys = buys
        self._data_iter = DataIterator(self._buys)

        self._trading_env = TradingSimulator()

        self.action_space = spaces.Discrete(2, )
        shape = self._get_initial_observation_state_shape()
        self.observation_space = spaces.Box(low=np.zeros(shape),
                                            high=np.ones(shape),
                                            dtype=np.float64)

    def step(self, actions):
        curr_seq = self._data_iter.curr_sequence

        reward = self._trading_env.step(actions, curr_seq)

        next_sequence = self._data_iter.next_sequence()
        next_state = self.forward_state(next_sequence)

        return next_state, reward, self._data_iter.episode_end, {}

    def close(self):
        pass

    def reset(self):
        self._data_iter.episode_end = False
        self._data_iter.episode_count += 1

        next_sequence = self._data_iter.next_sequence()
        state = self.forward_state(next_sequence)
        self._trading_env = TradingSimulator()
        return state


class EnvNN(BaseEnv):
    state_handler = StateHandlerNN(extend=False)


class EnvNNExtended(EnvNN):
    state_handler = StateHandlerNN(extend=True)


class EnvCNN(BaseEnv):
    state_handler = StateHandlerCNN(extend=False)


class EnvCNNExtended(EnvCNN):
    state_handler = StateHandlerCNN(extend=True)
