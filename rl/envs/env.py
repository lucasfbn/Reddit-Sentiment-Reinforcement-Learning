from collections import deque

import numpy as np
from tensorforce import Environment

from preprocessing.sequences import Sequence
from rl.envs.simple_trading import SimpleTradingEnvTraining
from rl.envs.utils.reward_counter import RewardCounter


class Env(Environment):

    def __init__(self, ticker):
        super().__init__()

        self.ticker = ticker
        self.ticker_iter = 0
        self.ticker_iter_max = len(self.ticker)

        self.curr_ticker = None
        self.curr_reward_counter = RewardCounter()
        self.curr_sequences = None
        self.curr_simple_trading_env = SimpleTradingEnvTraining("init")
        self.curr_sequence = None

        self.episode_end = False
        self.episode_count = 0

    @staticmethod
    def get_state(sequence: Sequence):
        raise NotImplementedError

    @staticmethod
    def shape_state(state):
        raise NotImplementedError

    def states(self):
        raise NotImplementedError

    def next_ticker(self):
        if self.ticker_iter == self.ticker_iter_max:
            self.ticker_iter = 0

        r = self.ticker[self.ticker_iter]

        self.ticker_iter += 1
        self.curr_ticker = r
        self.curr_sequences = deque(self.curr_ticker.sequences)

    def next_sequence(self):
        if len(self.curr_sequences) == 0:
            self.episode_end = True
        else:
            next_sequence = self.curr_sequences.popleft()
            self.curr_sequence = next_sequence

    def next_state(self):
        next_state = self.get_state(self.curr_sequence)
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
        price = self.curr_sequence.price

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

        self.next_sequence()
        next_state = self.next_state()
        return next_state, self.episode_end, reward

    def reset(self):
        self.episode_end = False
        self.episode_count += 1

        self.next_ticker()
        self.next_sequence()
        state = self.next_state()
        self.curr_simple_trading_env = SimpleTradingEnvTraining(ticker_name=self.curr_ticker.name)
        return state

    def log(self):
        self.curr_reward_counter.log(step=self.episode_count)
        self.curr_reward_counter = RewardCounter()

    def actions(self):
        return dict(type="int", num_values=3)

    # def max_episode_timesteps(self):
    #     return max(len(tck) for tck in self.ticker)


class EnvNN(Env):

    def states(self):
        shape = self.ticker[0].sequences[0].flat.shape
        return dict(type="float", shape=(shape[1],), min_value=0.0, max_value=1.0)

    @staticmethod
    def get_state(sequence: Sequence):
        return sequence.flat

    @staticmethod
    def shape_state(state):
        state = np.asarray(state).astype("float32")
        state = state.reshape((state.shape[1],))
        return state


class EnvCNN(Env):

    def states(self):
        shape = self.ticker[0].sequences[0].arr.shape
        return dict(type="float", shape=(1, shape[0], shape[1]), min_value=0.0, max_value=1.0)

    @staticmethod
    def get_state(sequence: Sequence):
        return sequence.arr

    @staticmethod
    def shape_state(state):
        state = state.values.reshape((1, state.shape[0], state.shape[1]))
        state = np.asarray(state).astype('float32')
        return state
