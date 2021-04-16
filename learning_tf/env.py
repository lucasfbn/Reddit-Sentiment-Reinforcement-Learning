from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

from collections import deque
from utils import log


class TradingEnv(py_environment.PyEnvironment):

    def __init__(self, data):
        super().__init__()

        self._episode_data = data
        self._data = deque(copy.deepcopy(data))
        self._x = None
        self._inventory = deque()
        self._end_of_timeseries = False
        self._episode_ended = False
        self._reward = 0

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=((72,)), dtype=np.float32, minimum=0,
                                                             name='observation')
        # self._observation_spec = array_spec.BoundedArraySpec(shape=((1, 7, 9)), dtype=np.float32, minimum=0,
        #                                                      name='observation')

        self._state = None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _calculate_margin(self, current_price):
        margin = 0

        for buy_price in self._inventory:
            margin += current_price - buy_price

        return margin

    def _handle_timeseries(self):

        if len(self._x) == 1 and len(self._data) == 1:
            self._end_of_timeseries = True
            self._episode_ended = True
        elif len(self._x) > 1:
            self._end_of_timeseries = False
        elif len(self._x) == 1:
            self._end_of_timeseries = True
        else:
            raise ValueError("Should not happen.")

    def _step(self, action):

        if self._end_of_timeseries:
            self._next_timeseries()

        current_price = self._current_price()

        if action == 1:
            self._inventory.append(current_price)
            log.debug(f"Buy: {current_price}, len inventory: {len(self._inventory)}")

        elif action == 2:

            if len(self._inventory) > 0:

                margin = self._calculate_margin(current_price)
                self._reward += margin
                log.debug(f"Added to reward: {margin}. Total reward: {self._reward}")

                self._inventory = deque()
                log.debug(f"Sell: {current_price}, Profit: {margin}")
            else:
                log.debug(f"Attempted sell, but inventory is empty.")

        self._handle_timeseries()

        if self._episode_ended:
            return ts.termination(self._state, self._reward)
        else:
            current_state = self._state
            self._state = self._shape_state(self._x.popleft())
            return ts.transition(self._state, reward=self._reward, discount=1.0)

    def _next_timeseries(self):
        self._x = self._data.popleft()["data"]
        self._x = deque(self._x)

        self._inventory = deque()

        self._state = self._x.popleft()
        self._state = self._shape_state(self._state)

    def _reset(self):
        self._end_of_timeseries = False
        self._episode_ended = False

        self._data = deque(copy.deepcopy(self._episode_data))

        self._x = self._data.popleft()["data"]
        self._x = deque(self._x)

        self._inventory = deque()
        self._reward = 0

        self._state = self._x.popleft()
        self._state = self._shape_state(self._state)

        return ts.restart(self._state)


class EnvNN(TradingEnv):

    def _shape_state(self, state):
        state = np.asarray(state).astype("float32")
        state = state.reshape((state.shape[1],))
        return state

    def _current_price(self):
        shape = self._state.shape
        last_element = self._state[shape[0] - 1]
        return last_element


class EnvCNN(TradingEnv):

    def _shape_state(self, state):
        state = state.values.reshape((1, state.shape[0], state.shape[1]))
        state = np.asarray(state).astype('float32')
        return state

    def _current_price(self):
        shape = self._state.shape
        last_row = self._state[0][shape[1] - 1]
        last_element = last_row[shape[2] - 1]
        return last_element


if __name__ == '__main__':
    import paths
    import pickle as pkl

    with open(paths.datasets_data_path / "_0" / "timeseries.pkl", "rb") as f:
        data = pkl.load(f)
    # data = data[:2]

    ev = EnvNN(data)

    # print(utils.validate_py_environment(ev, episodes=5))

    ev._reset()
    ev._step(0)
    #
    # for _ in range(20):
    #     ev._step(1)
    #
    # ev._reset()
