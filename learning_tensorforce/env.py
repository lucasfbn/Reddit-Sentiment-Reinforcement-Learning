import copy
from collections import deque
from random import shuffle

import mlflow
import numpy as np
from tensorforce import Environment

from utils.mlflow_api import load_file
from utils.util_funcs import log


class Env(Environment):
    data = None

    def __init__(self):
        super().__init__()

        self._episode_data = self.data

        self._inventory = deque()
        self._episode_reward = 0

        self._episode_ended = False
        self._x = None
        self._state = None

        # Counter
        self._counter = 0
        self._episode_counter = 0
        self._neg_reward_counter = 0
        self._even_reward_counter = 0
        self._pos_reward_counter = 0

    def set_data(self, data):
        self.data = data

    def _shape_state(self, state):
        raise NotImplemented

    def _current_price(self):
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

        next_state = self._shape_state(self._x.popleft())

        if len(self._x) == 0:
            self._episode_ended = True

        self._handle_step_counter(reward)
        self._state = next_state

        return next_state, self._episode_ended, reward

    def _handle_reset_counter(self):
        mlflow.log_metric("full run reward", self._episode_reward, step=self._episode_counter)
        mlflow.log_metric("n_even_rewards", self._even_reward_counter, step=self._episode_counter)
        mlflow.log_metric("n_pos_rewards", self._pos_reward_counter, step=self._episode_counter)
        mlflow.log_metric("n_neg_rewards", self._neg_reward_counter, step=self._episode_counter)

        self._episode_reward = 0
        self._even_reward_counter = 0
        self._neg_reward_counter = 0
        self._pos_reward_counter = 0

    def reset(self):
        self._episode_ended = False

        self._x = copy.deepcopy(self._episode_data[self._counter].array_sequence)
        shuffle(self._x)
        self._counter += 1
        self._episode_counter += 1

        if self._counter >= len(self._episode_data):
            self._counter = 0
            self._handle_reset_counter()

        mlflow.log_metric("counter", self._counter, step=self._episode_counter)

        self._x = deque(self._x)

        self._inventory = deque()

        try:
            self._state = self._x.popleft()
        except:
            pass
        self._state = self._shape_state(self._state)

        return self._state

    def max_episode_timesteps(self):

        max_ = 0

        for d in self.data:
            assert len(d.array_sequence) == len(d.flat_sequence)
            if len(d.array_sequence) > max_:
                max_ = len(d.array_sequence)

        return max_

    def actions(self):
        return dict(type="int", num_values=3)


class EnvNN(Env):

    def states(self):
        shape = self.data[0].flat_sequence[0].shape
        return dict(type="float", shape=(shape[1],))

    def _shape_state(self, state):
        state = np.asarray(state).astype("float32")
        state = state.reshape((state.shape[1],))
        return state

    def _current_price(self):
        shape = self._state.shape
        last_element = self._state[shape[0] - 1]
        return last_element


class EnvCNN(Env):

    def states(self):
        shape = self.data[0].array_sequence[0].shape
        return dict(type="float", shape=(1, shape[0], shape[1]))

    def _shape_state(self, state):
        state = state.values.reshape((1, state.shape[0], state.shape[1]))
        state = np.asarray(state).astype('float32')
        return state

    def _current_price(self):
        shape = self._state.shape
        last_row = self._state[0][shape[1] - 1]
        last_element = last_row[shape[2] - 1]
        return last_element


if __name__ == "__main__":
    import paths
    import random

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Testing-Environment")  #

    with mlflow.start_run():
        data = load_file(run_id="031bfd42e667441cb4a54056f11d60b2", fn="ticker.pkl", experiment="Tests")
        EnvCNN.data = data

        env = EnvCNN()

        for _ in range(10000):
            states = env.reset()
            terminal = False
            print(_)
            states_ = []

            while not terminal:
                actions = random.randint(0, 2)
                states, terminal, reward = env.execute(actions=actions)
