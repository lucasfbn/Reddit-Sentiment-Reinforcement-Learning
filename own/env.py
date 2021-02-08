from collections import deque
from gym import Env
from gym.spaces import Discrete


class StockEnv:

    def __init__(self):
        self._state = None

        self._df = None
        self._prices_raw = None
        self._inventory = deque()
        self.total_profit = 0

    def _calculate_margin(self, current_price):
        margin = 0

        for buy_price in self._inventory:
            margin += current_price - buy_price

        return margin

    def step(self, action):

        current_price = self._state[len(self._state) - 1]

        reward = 0

        if action == 1:
            self._inventory.append(current_price)
            print(f"Buy: {current_price}, len inventory: {len(self._inventory)}")

        elif action == 2:

            if len(self._inventory) > 0:

                margin = self._calculate_margin(current_price)
                self.total_profit += margin

                reward = max(margin, 0)

                self._inventory = deque()
                print(f"Sell: {current_price}, Profit: {margin}")

            else:
                print(f"Attempted sell, but inventory is empty.")

        done = True if len(self._df) == 1 else False

        # Add information whether we have an inventory or not to the state
        if len(self._inventory) == 0:
            next_state = [0] + self._df.popleft()
        else:
            next_state = [1] + self._df.popleft()

        self._state = next_state

        return next_state, reward, done, None

    def _convert_df(self, df):
        return deque(df.values.tolist())

    def reset(self, df):
        self._df = self._convert_df(df)
        self._inventory = deque()
        self.total_profit = 0

        self._state = [0] + self._df.popleft()
        return self._state
