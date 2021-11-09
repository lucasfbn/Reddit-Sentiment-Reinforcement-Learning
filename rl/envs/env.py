from collections import deque

import numpy as np
from tensorforce import Environment

from preprocessing.sequences import Sequence
from rl.envs.simple_trading import SimpleTradingEnvTraining
from rl.envs.utils.reward_counter import RewardCounter
from rl.envs.utils.data_iterator import DataIterator


class Env(Environment):

    def __init__(self, ticker):
        super().__init__()

        self.data_iter = DataIterator(ticker)

        self.curr_reward_counter = RewardCounter()
        self.curr_simple_trading_env = SimpleTradingEnvTraining("init")

    @staticmethod
    def get_state_field(sequence: Sequence):
        """
        Implemented in subclasses. Determines which field of the sequence object
        contains the state.
        """
        raise NotImplementedError

    @staticmethod
    def shape_state(state):
        raise NotImplementedError

    def states(self):
        raise NotImplementedError

    def next_state(self, sequence):
        next_state = self.get_state_field(sequence)
        next_state = self.shape_state(next_state)
        return next_state

    def hold(self, price):
        reward = self.curr_simple_trading_env.hold(price)
        return reward

    def buy(self, price):
        reward = self.curr_simple_trading_env.buy(price)
        return reward

    def sell(self, price):
        reward = self.curr_simple_trading_env.sell(price)
        return reward

    def execute(self, actions):
        price = self.data_iter.curr_sequence.price

        # Hold
        if actions == 0:
            reward = self.hold(price)

        # Buy
        elif actions == 1:
            reward = self.buy(price)

        # Sell
        elif actions == 2:
            reward = self.sell(price)

        else:
            raise ValueError("Invalid action.")

        self.curr_reward_counter.add_reward(reward)

        next_sequence = self.data_iter.next_sequence()
        next_state = self.next_state(next_sequence)
        return next_state, self.data_iter.is_episode_end(), reward

    def reset(self):
        self.data_iter.episode_end = False
        self.data_iter.episode_count += 1

        self.data_iter.next_ticker()
        next_sequence = self.data_iter.next_sequence()
        state = self.next_state(next_sequence)
        self.curr_simple_trading_env = SimpleTradingEnvTraining(ticker_name=self.data_iter.curr_ticker.name)
        return state

    def log(self):
        self.curr_reward_counter.log(step=self.data_iter.episode_count)
        self.curr_reward_counter = RewardCounter()

    def actions(self):
        return dict(type="int", num_values=3)

    # def max_episode_timesteps(self):
    #     return max(len(tck) for tck in self.ticker)


class EnvNN(Env):

    def states(self):
        shape = self.data_iter.ticker[0].sequences[0].flat.shape
        return dict(type="float", shape=(shape[1],), min_value=0.0, max_value=1.0)

    @staticmethod
    def get_state_field(sequence: Sequence):
        return sequence.flat

    @staticmethod
    def shape_state(state):
        state = np.asarray(state).astype("float32")
        state = state.reshape((state.shape[1],))
        return state


class EnvCNN(Env):

    def states(self):
        shape = self.data_iter.ticker[0].sequences[0].arr.shape
        return dict(type="float", shape=(1, shape[0], shape[1]), min_value=0.0, max_value=1.0)

    @staticmethod
    def get_state_field(sequence: Sequence):
        return sequence.arr

    @staticmethod
    def shape_state(state):
        state = state.values.reshape((1, state.shape[0], state.shape[1]))
        state = np.asarray(state).astype('float32')
        return state
