from collections import deque
from abc import ABC, abstractmethod
import numpy as np

verbose = False

if verbose:
    def vprint(msg):
        print(msg)
else:
    def vprint(msg):
        pass


class Env(ABC):

    def __init__(self):
        self._state = None

        self._x = None
        self._inventory = deque()
        self.total_profit = 0

    def _calculate_margin(self, current_price):
        margin = 0

        for buy_price in self._inventory:
            margin += current_price - buy_price

        return margin

    def step(self, action):

        reward = 0

        current_price = self._current_price()

        if action == 1:
            self._inventory.append(current_price)
            vprint(f"Buy: {current_price}, len inventory: {len(self._inventory)}")

        elif action == 2:

            if len(self._inventory) > 0:

                margin = self._calculate_margin(current_price)
                self.total_profit += margin

                reward = max(margin, 0)

                self._inventory = deque()
                vprint(f"Sell: {current_price}, Profit: {margin}")

            else:
                vprint(f"Attempted sell, but inventory is empty.")

        done = True if len(self._x) == 0 else False

        if not done:
            next_state = self._shape_state(self._x.popleft())
        else:
            next_state = None
        self._state = next_state

        return next_state, reward, done, None

    def reset(self, x):
        self._x = deque(x)
        self._inventory = deque()
        self.total_profit = 0

        self._state = self._x.popleft()
        self._state = self._shape_state(self._state)
        return self._state

    @abstractmethod
    def _shape_state(self, state):
        pass

    @abstractmethod
    def _current_price(self):
        pass


class Env_NN(Env):

    def _shape_state(self, state):
        return np.array(state)

    def _current_price(self):
        shape = self._state.shape
        last_element = self._state[0][shape[1] - 1]
        return last_element


class Env_CNN(Env):

    def _shape_state(self, state):
        state = state.values.reshape((1, state.shape[0], state.shape[1]))
        return state

    def _current_price(self):
        shape = self._state.shape
        last_row = self._state[0][shape[1] - 1]
        last_element = last_row[shape[2] - 1]
        return last_element
