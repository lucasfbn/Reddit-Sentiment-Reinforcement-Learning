class Inventory:

    def __init__(self):
        self._inv = []
        self._day = 0

    def add(self, sequence):
        self._inv.append({"seq": sequence, "removal_day": self._day + sequence.evl.days_cash_bound})

    def inventory_state(self, sequence):
        return int(any(sequence.metadata.ticker_name == seq["seq"].metadata.ticker_name for seq in self._inv))

    def _update(self):
        removed = []
        new_inv = []

        for item in self._inv:
            if item["removal_day"] == self._day:
                removed.append(item)
            else:
                new_inv.append(item)

        self._inv = new_inv

        # 1 + ... because the reward only captures the profit. When we sell, however,
        # we also get the initial inset, which is either higher or lower, depending on
        # the profit
        return sum((1 + item["seq"].evl.reward_backtracked) for item in removed)

    def new_day(self):
        self._day += 1
        return self._update()


class TradingSimulator:
    N_START_TRADES = 20

    def __init__(self):
        self._inventory = Inventory()
        self._n_trades = self.N_START_TRADES

    @property
    def inventory(self):
        return self._inventory

    @property
    def n_trades_left_scaled(self):
        return self._n_trades / self.N_START_TRADES if self._n_trades <= self.N_START_TRADES else 1.0

    def new_day(self):
        self._n_trades += self._inventory.new_day()

    def step(self, action, sequence):

        if action == 0:
            reward = 0

        elif action == 1:

            reward = sequence.evl.reward_backtracked

            if self._n_trades < 1:
                reward *= -1 if reward > 0 else 2
            else:
                self._inventory.add(sequence)
                self._n_trades -= 1

        else:
            raise ValueError("Invalid action.")

        return reward
