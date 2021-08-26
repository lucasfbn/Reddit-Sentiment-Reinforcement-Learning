from collections import deque

import mlflow
import numpy as np
from tensorforce import Environment

from utils.util_funcs import log


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
    ENABLE_TRANSACTION_COSTS = True
    TRANSACTION_COSTS_PERC = 0.01

    ENABLE_NEG_BUY_REWARD = True

    def __init__(self, ticker):
        super().__init__()

        self.ticker = ticker
        self.ticker_iter = 0
        self.ticker_iter_max = len(self.ticker)

        self.curr_ticker = None
        self.curr_env_counter = EnvCounter()
        self.curr_sequences = None
        self.curr_inventory = []
        self.curr_sequence = None

        self._episode_end = False
        self._episode_count = 0

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
            self._episode_end = True
            return None
        else:
            next_sequence = self.curr_sequences.popleft()
            next_sequence = self.shape_state(next_sequence)
            next_state = next_sequence.df
            self.curr_sequence = next_sequence
            return next_state

    def calculate_margin(self, current_price):
        margin = 0

        for buy_price in self.curr_inventory:
            margin += current_price - buy_price

        return margin

    def hold(self, reward, price):
        return reward

    def buy(self, reward, price):
        self.curr_inventory.append(price)

        if self.ENABLE_NEG_BUY_REWARD:
            reward -= price

        if self.ENABLE_TRANSACTION_COSTS:
            reward -= price * self.TRANSACTION_COSTS_PERC

        log.debug(f"BUY. Stock: {self.curr_ticker.name}. Relativ price: {self.curr_sequence.price}"
                  f"Abs price: {self.curr_sequence.price_raw}")

        return reward

    def sell(self, reward, price):

        if len(self.curr_inventory) > 0:

            margin = self.calculate_margin(price)

            if self.ENABLE_TRANSACTION_COSTS:
                margin += margin * self.TRANSACTION_COSTS_PERC

            reward += margin

            self.curr_inventory = []

            log.debug(f"SOLD: Stock: {self.curr_ticker.name}. Relativ price: {self.curr_sequence.price}"
                      f"Abs price: {self.curr_sequence.price_raw}. Profit/Loss: {margin}")
        else:
            log.debug(f"Attempted sell, but inventory is empty.")

        return reward

    def execute(self, actions):
        reward = 0
        price = self.curr_sequence.price

        # Hold
        if actions == 0:
            reward = self.hold(reward, price)

        # Buy
        elif actions == 1:
            reward = self.buy(reward, price)

        # Sell
        elif actions == 2:
            reward = self.sell(reward, price)

        self.curr_env_counter.add_reward(reward)

        next_state = self.next_sequence()
        return next_state, self._episode_end, reward

    def reset(self):
        self._episode_end = False
        self._episode_count += 1

        self.curr_inventory = []
        self.next_ticker()
        state = self.next_sequence()
        return state

    def log(self):
        self.curr_env_counter.log(step=self._episode_count)
        self.curr_env_counter = EnvCounter()

    def actions(self):
        return dict(type="int", num_values=3)

    def max_episode_timesteps(self):
        return max(len(tck) for tck in self.ticker)


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
