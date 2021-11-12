from abc import ABC

import numpy as np
from gym import Env, spaces

from preprocessing.sequences import Sequence
from rl.train.envs.sub_envs.trading import SimpleTradingEnvTraining
from rl.train.envs.utils.data_iterator import DataIterator
from rl.train.envs.utils.reward_counter import RewardCounter
from rl.train.envs.utils.state_extender import StateExtenderNN, StateExtenderCNN


class BaseEnv(Env):

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

    def _get_first_sequence(self):
        return self.data_iter.ticker[0].sequences[0]

    def _get_initial_observation_state_shape(self):
        return self.get_state(self._get_first_sequence()).shape

    @staticmethod
    def get_state(sequence):
        """
        Implemented in subclasses. Determines which field of the sequence object
        contains the state.
        """
        raise NotImplementedError

    @staticmethod
    def shape_state(sequence):
        raise NotImplementedError

    def next_state(self, sequence):
        next_state = self.get_state(sequence)
        next_state = self.shape_state(next_state)
        return next_state

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
        next_state = self.next_state(next_sequence)

        return next_state, reward, self.data_iter.is_episode_end(), {}

    def close(self):
        pass

    def reset(self):
        self.data_iter.episode_end = False
        self.data_iter.episode_count += 1

        self.data_iter.next_ticker()
        next_sequence = self.data_iter.next_sequence()
        state = self.next_state(next_sequence)
        self.trading_env = SimpleTradingEnvTraining(ticker_name=self.data_iter.curr_ticker.name)
        return state

    def log(self):
        self.reward_counter.log(step=self.data_iter.episode_count)
        self.reward_counter = RewardCounter()


class StateExtenderEnv(BaseEnv):
    state_extender = None

    def _extend_state(self, state):
        inventory_state = 1 if len(self.trading_env.inventory) > 0 else 0
        extended_state = self.state_extender.add_inventory_state(state, inventory_state)

    def next_state(self, sequence):
        next_state = self.get_state(sequence)
        next_state = self._extend_state(next_state)
        next_state = self.shape_state(next_state)
        return next_state

    def _get_initial_observation_state_shape(self):
        shape = self.get_state(self._get_first_sequence()).shape
        return self.state_extender.get_new_shape_state(shape)


class EnvNN(BaseEnv):

    @staticmethod
    def get_state(sequence: Sequence):
        return sequence.flat

    @staticmethod
    def shape_state(state):
        state = np.asarray(state).astype("float32")
        state = state.reshape((state.shape[1],))
        return state


class EnvNNExtended(EnvNN, StateExtenderEnv):
    state_extender = StateExtenderNN


class EnvCNN(BaseEnv):

    @staticmethod
    def get_state(sequence: Sequence):
        return sequence.arr

    @staticmethod
    def shape_state(state):
        state = state.values.reshape((1, state.shape[0], state.shape[1]))
        state = np.asarray(state).astype('float32')
        return state


class EnvCNNExtended(EnvCNN, StateExtenderEnv):
    state_extender = StateExtenderCNN
