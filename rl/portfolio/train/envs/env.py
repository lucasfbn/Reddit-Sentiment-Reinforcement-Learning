from abc import ABC, abstractmethod

import numpy as np
from gym import Env, spaces

from preprocessing.sequence import Sequence
from rl.portfolio.train.envs.sub_envs.trading import TradingSimulator
from rl.portfolio.train.envs.utils.data_iterator import DataIterator
from rl.utils.state_handler import StateHandlerCNN, StateHandlerNN


class BaseEnv(Env, ABC):

    def __init__(self, buys):
        super().__init__()

        self._buys = buys
        self._sequence_iter = DataIterator(self._buys).sequence_iter()

        self._trading_env = TradingSimulator()

        self.action_space = spaces.Discrete(2, )
        shape = self._get_initial_observation_state_shape()
        self.observation_space = spaces.Box(low=np.zeros(shape),
                                            high=np.ones(shape),
                                            dtype=np.float64)

    @property
    @abstractmethod
    def state_handler(self):
        pass

    def forward_state(self, sequence: Sequence):
        inventory_state = self._trading_env.inventory.inventory_state(sequence)
        probability = sequence.evl.buy_proba
        n_trades_left = self._trading_env.n_trades_left_scaled
        return self.state_handler.forward(sequence, [inventory_state, probability, n_trades_left])

    def _get_first_sequence(self):
        return self._sequence_iter.sequences[0]

    def _get_initial_observation_state_shape(self):
        return self.forward_state(self._get_first_sequence()).shape

    def step(self, actions):
        curr_seq = self._sequence_iter.curr_sequence

        reward = self._trading_env.step(actions, curr_seq)

        next_sequence, is_episode_end, is_new_date = next(self._sequence_iter)

        if is_new_date:
            self._trading_env.new_day()

        next_state = self.forward_state(next_sequence)

        return next_state, reward, is_episode_end, {}

    def close(self):
        pass

    def reset(self):
        self._sequence_iter.episode_end = False
        self._sequence_iter.episode_count += 1

        next_sequence, _, _ = next(self._sequence_iter)
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
