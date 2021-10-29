from collections import deque

import mlflow
import numpy as np
from tensorforce import Environment

from rl.envs.simple_trading import SimpleTradingEnv


class EnvCounter:

    def __init__(self):
        self.full = 0
        self.neg = 0
        self.even = 0
        self.pos = 0

    def add_reward(self, reward):
        self.full += reward

        if reward == 0.0:
            self.even += 1
        elif reward > 0.0:
            self.pos += 1
        elif reward < 0.0:
            self.neg += 1
        else:
            raise ValueError("Reward")

    def log(self, step):
        mlflow.log_metric("full run reward", self.full, step=step)
        mlflow.log_metric("n_even_rewards", self.even, step=step)
        mlflow.log_metric("n_pos_rewards", self.pos, step=step)
        mlflow.log_metric("n_neg_rewards", self.neg, step=step)


class Env(Environment):

    def __init__(self, ticker):
        super().__init__()

        self.ticker = ticker
        self.ticker_iter = 0
        self.ticker_iter_max = len(self.ticker)

        self.curr_ticker = None
        self.curr_env_counter = EnvCounter()
        self.curr_sequences = None
        self.curr_simple_trading_env = SimpleTradingEnv("init")
        self.curr_sequence = None

        self.episode_end = False
        self.episode_count = 0

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
            current_state = self.curr_sequence.df
            return current_state
        else:
            next_sequence = self.curr_sequences.popleft()
            next_sequence = self.shape_state(next_sequence)
            next_state = next_sequence.df
            self.curr_sequence = next_sequence
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

        self.curr_env_counter.add_reward(reward)

        next_state = self.next_sequence()
        return next_state, self.episode_end, reward

    def reset(self):
        self.episode_end = False
        self.episode_count += 1

        self.next_ticker()
        state = self.next_sequence()
        self.curr_simple_trading_env = SimpleTradingEnv(ticker_name=self.curr_ticker.name)
        return state

    def log(self):
        self.curr_env_counter.log(step=self.episode_count)
        self.curr_env_counter = EnvCounter()

    def actions(self):
        return dict(type="int", num_values=3)

    # def max_episode_timesteps(self):
    #     return max(len(tck) for tck in self.ticker)


class EnvNN(Env):

    def states(self):
        shape = self.ticker[0].sequences[0].flat.shape
        return dict(type="float", shape=(shape[1],))

    @staticmethod
    def shape_state(state):
        state.df = np.asarray(state.flat).astype("float32")
        state.df = state.df.reshape((state.df.shape[1],))
        return state


class EnvCNN(Env):

    def states(self):
        shape = self.ticker[0].sequences[0].arr.shape
        return dict(type="float", shape=(1, shape[0], shape[1]))

    @staticmethod
    def shape_state(state):
        state.df = state.arr.values.reshape((1, state.arr.shape[0], state.arr.shape[1]))
        state.df = np.asarray(state.df).astype('float32')
        return state
