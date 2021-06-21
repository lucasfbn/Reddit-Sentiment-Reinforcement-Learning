import copy
from collections import deque
from random import shuffle

import mlflow
import numpy as np
from tensorforce import Environment

from utils.util_funcs import log


class Env(Environment):
    data = None
    shuffle_sequences = True

    def __init__(self):
        super().__init__()

        self._episode_data = self.data

        self._inventory = deque()
        self._episode_reward = 0

        self._current_ticker = None
        self._current_sequences = None
        self._states = None

        self._episode_ended = False
        self._state = None

        # Counter
        self._counter = 0
        self._episode_counter = 0
        self._neg_reward_counter = 0
        self._even_reward_counter = 0
        self._pos_reward_counter = 0

    def _shape_state(self, state):
        raise NotImplemented

    def _calculate_margin(self, current_price):
        margin = 0

        for buy_price in self._inventory:
            margin += current_price - buy_price

        return margin

    def _handle_step_counter(self, reward):
        self._episode_reward += reward

        if reward == 0.0:
            self._even_reward_counter += 1
        elif reward > 0.0:
            self._pos_reward_counter += 1
        elif reward < 0.0:
            self._neg_reward_counter += 1
        else:
            raise ValueError("Reward")

    def execute(self, actions):

        reward = 0
        current_price = self._current_price()

        if actions == 1:
            self._inventory.append(current_price)
            log.debug(f"Buy: {current_price}, len inventory: {len(self._inventory)}")

        elif actions == 2:

            if len(self._inventory) > 0:

                margin = self._calculate_margin(current_price)
                reward += margin
                log.debug(f"Added to reward: {margin}. Total reward: {reward}")

                self._inventory = deque()
                log.debug(f"Sell: {current_price}, Profit: {margin}")
            else:
                log.debug(f"Attempted sell, but inventory is empty.")

        if len(self._states) == 0:
            self._episode_ended = True
        else:
            next_state = self._shape_state(self._states.popleft())
            self._state = next_state

        self._handle_step_counter(reward)

        return self._state.df, self._episode_ended, reward

    def _handle_reset_counter(self):
        mlflow.log_metric("full run reward", self._episode_reward, step=self._episode_counter)
        mlflow.log_metric("n_even_rewards", self._even_reward_counter, step=self._episode_counter)
        mlflow.log_metric("n_pos_rewards", self._pos_reward_counter, step=self._episode_counter)
        mlflow.log_metric("n_neg_rewards", self._neg_reward_counter, step=self._episode_counter)

        self._episode_reward = 0
        self._even_reward_counter = 0
        self._neg_reward_counter = 0
        self._pos_reward_counter = 0

    def _handle_counter(self):
        self._counter += 1
        self._episode_counter += 1

        if self._counter >= len(self._episode_data):
            self._counter = 0
            self._handle_reset_counter()

        mlflow.log_metric("counter", self._counter, step=self._episode_counter)

    def _assign_new_ticker(self):
        self._current_ticker = copy.deepcopy(self._episode_data[self._counter])

    def _assign_new_sequences(self):
        self._current_sequences = self._current_ticker.sequences

    def _assign_new_states(self):
        if self.shuffle_sequences:
            shuffle(self._current_sequences)
        self._states = deque(self._current_sequences)

    def reset(self):
        self._episode_ended = False

        self._assign_new_ticker()
        self._assign_new_sequences()
        self._assign_new_states()

        self._handle_counter()
        self._inventory = deque()

        self._state = self._states.popleft()
        self._state = self._shape_state(self._state)

        return self._state.df

    def max_episode_timesteps(self):

        max_ = 0

        for ticker in self.data:
            if len(ticker.sequences) > max_:
                max_ = len(ticker.sequences)

        return max_

    def actions(self):
        return dict(type="int", num_values=3)

    def _current_price(self):
        return self._state.price

    @staticmethod
    def get_sequences(ticker):
        return ticker.sequences


class EnvNN(Env):

    def states(self):
        shape = self.data[0].sequence[0].flat.shape
        return dict(type="float", shape=(shape[1],))

    @staticmethod
    def _shape_state(state):
        state.df = np.asarray(state.flat).astype("float32")
        state.df = state.df.reshape((state.df.shape[1],))
        return state


class EnvCNN(Env):

    def states(self):
        shape = self.data[0].sequence[0].arr.shape
        return dict(type="float", shape=(1, shape[0], shape[1]))

    @staticmethod
    def _shape_state(state):
        state.df = state.arr.values.reshape((1, state.arr.shape[0], state.arr.shape[1]))
        state.df = np.asarray(state.df).astype('float32')
        return state
