from collections import deque

verbose = False

if verbose:
    def vprint(msg):
        print(msg)
else:
    def vprint(msg):
        pass


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

        done = True if len(self._df) == 1 else False

        next_state = self._df.popleft()
        self._state = next_state

        return next_state, reward, done, None

    def _convert_df(self, df):
        return deque(df.values.tolist())

    def reset(self, df):
        self._df = self._convert_df(df)
        self._inventory = deque()
        self.total_profit = 0

        self._state = self._df.popleft()
        return self._state