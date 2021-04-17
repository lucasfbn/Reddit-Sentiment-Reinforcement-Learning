import copy
from collections import deque

import mlflow
import numpy as np
from tensorforce import Environment

from utils import log


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

    def states(self):
        raise NotImplemented

    def actions(self):
        raise NotImplemented

    def max_episode_timesteps(self):
        raise NotImplemented

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

        self._x = copy.deepcopy(self._episode_data[self._counter]["data"])
        self._counter += 1
        self._episode_counter += 1

        if self._counter >= len(self._episode_data):
            self._counter = 0
            self._handle_reset_counter()

        mlflow.log_metric("counter", self._counter, step=self._episode_counter)

        self._x = deque(self._x)

        self._inventory = deque()

        self._state = self._x.popleft()
        self._state = self._shape_state(self._state)

        return self._state


class EnvNN(Env):

    def states(self):
        return dict(type="float", shape=(72,))

    def actions(self):
        return dict(type="int", num_values=3)

    def max_episode_timesteps(self):
        return 53

    def _shape_state(self, state):
        state = np.asarray(state).astype("float32")
        state = state.reshape((state.shape[1],))
        return state

    def _current_price(self):
        shape = self._state.shape
        last_element = self._state[shape[0] - 1]
        return last_element


if __name__ == "__main__":
    import paths
    import pickle as pkl
    import random

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Testing")  #
    mlflow.start_run()

    with open(paths.datasets_data_path / "_0" / "timeseries.pkl", "rb") as f:
        data = pkl.load(f)

    EnvNN.data = data

    env = EnvNN()

    for _ in range(10000):
        states = env.reset()
        terminal = False

        states_ = []

        while not terminal:
            actions = random.randint(0, 2)
            states, terminal, reward = env.execute(actions=actions)

    mlflow.end_run()
