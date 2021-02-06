from collections import deque
from gym import Env
from gym.spaces import Discrete


class StockEnv:

    def __init__(self, eval=False):
        self.eval = eval
        self.action_space = 3
        self.observation_space = 24

        self._state = None

        self._df = None
        self._prices_raw = None
        self._inventory = deque()
        self.total_profit = 0

    def _calculate_margin(self, current_price):

        if self.eval:
            margin = 1
        else:
            margin = 0

        for buy_price in self._inventory:

            if self.eval:
                margin *= current_price / (buy_price + 0.01)
            else:
                margin += current_price - buy_price

        self._inventory = deque()

        return margin

    def step(self, action):

        if self.eval:
            current_price = self._prices_raw.popleft()
            assert len(self._prices_raw) == len(self._df)
        else:
            current_price = self._state[len(self._state) - 1]

        reward = 0

        if action == 1:
            self._inventory.append(current_price)
            print(f"Buy: {current_price}, len inventory: {len(self._inventory)}")

        elif action == 2:

            if len(self._inventory) > 0:

                margin = self._calculate_margin(current_price)

                if self.eval:
                    self.total_profit *= margin
                else:
                    self.total_profit += margin

                reward = max(margin, 0)
                print(f"Sell: {current_price}, Profit: {margin}")

            else:
                print(f"Attempted sell, but inventory is empty.")

        done = True if len(self._df) == 1 else False

        next_state = self._df.popleft()
        self._state = next_state

        return next_state, reward, done, None

    def _convert_df(self, df):
        return deque(df.values.tolist())

    def reset(self, df, prices_raw=None):
        self._df = self._convert_df(df)
        self._inventory = deque()
        self.total_profit = 0

        if self.eval:
            self._prices_raw = deque(prices_raw.values.tolist())
            self.total_profit = 1

        self._state = self._df.popleft()
        return self._state
