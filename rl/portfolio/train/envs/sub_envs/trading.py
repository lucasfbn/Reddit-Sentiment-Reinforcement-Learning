import logging
from math import e

log = logging.getLogger("root")


class Inventory:

    def __init__(self):
        self._inv = []
        self._day = 0

    def inv_len(self):
        return len(self._inv)

    def add(self, sequence):
        log.debug(f"Added seq {sequence.metadata.ticker_name} to inv "
                  f"Removal date: {self._day + sequence.evl.days_cash_bound}")
        self._inv.append({"seq": sequence, "removal_day": self._day + sequence.evl.days_cash_bound})

    def inventory_state(self, sequence):
        inv_state = int(any(sequence.metadata.ticker_name == seq["seq"].metadata.ticker_name for seq in self._inv))
        log.debug(f"Inv state for seq: {sequence.metadata.ticker_name}: {inv_state}")
        return inv_state

    def _update(self):
        log.debug("Started inventory update")

        removed = []
        new_inv = []

        for item in self._inv:
            if item["removal_day"] == self._day:
                log.debug(f"\t Seq {item['seq'].metadata.ticker_name} is removed")
                removed.append(item)
            else:
                new_inv.append(item)

        self._inv = new_inv

        # 1 + ... because the reward only captures the profit. When we sell, however,
        # we also get the initial inset, which is either higher or lower, depending on
        # the profit
        total_reward = sum((1 + item["seq"].evl.reward_backtracked) for item in removed)
        log.debug(f"\t Total reward of inventory update: {total_reward}")
        return total_reward

    def new_day(self):
        log.debug("Called new day")
        self._day += 1
        return self._update()


class TradingSimulator:
    N_START_TRADES = 40

    def __init__(self):
        self._inventory = Inventory()
        self._n_trades = self.N_START_TRADES

    @property
    def n_trades(self):
        return self._n_trades

    @property
    def inventory(self):
        return self._inventory

    @property
    def n_trades_left_scaled(self):
        return self._n_trades / self.N_START_TRADES

    def trades_exhausted(self):
        return self._n_trades < 1

    def new_day(self):
        self._n_trades += self._inventory.new_day()

    def step(self, action, sequence):

        success = True

        if action == 0:
            log.debug("Action == 0")
            reward = 0.0

        elif action == 1:
            log.debug("Action == 1")

            reward = sequence.evl.reward_backtracked

            if self._n_trades < 1:
                success = False
            else:
                self._inventory.add(sequence)
                self._n_trades -= 1

        else:
            raise ValueError("Invalid action.")

        return reward, success
