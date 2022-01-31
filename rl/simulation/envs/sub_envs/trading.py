from dataclasses import dataclass


@dataclass
class Stock:
    kind: str
    ticker: str
    price: float
    quantity: int


class TradingSimulator:
    START_BALANCE = 2000
    INVESTMENT_PER_TRADE = 50
    MAX_PRICE_PER_STOCK = None
    SLIPPAGE = 0.007
    ORDER_FEE = 0.02

    def __init__(self):
        self._inventory = []
        self._balance = self.START_BALANCE
        self._n_trades = 0
        self._additional_costs_perc = self.SLIPPAGE + self.ORDER_FEE

    def reset_day(self):
        self._n_trades = 0

    @staticmethod
    def _base_checks(sequence):
        return sequence.metadata.tradeable

    def buy(self, sequence):
        if not self._base_checks(sequence):
            return False

        if self._balance <= 0 or (self._balance - self.INVESTMENT_PER_TRADE) <= 0:
            return False

        price = sequence.metadata.price_raw

        if self.MAX_PRICE_PER_STOCK is not None and price > self.MAX_PRICE_PER_STOCK:
            return False

        price = price * (1 + self._additional_costs_perc)
        quantity = self.INVESTMENT_PER_TRADE // price

        self._inventory.append(Stock(kind="buy", ticker=sequence.metadata.ticker_name, price=price, quantity=quantity))
        self._n_trades += 1
        self._balance -= price * quantity

        return True

    def sell(self, sequence):

        if not self._base_checks(sequence):
            return False

        new_inventory = []

        for stock in self._inventory:

            if stock.ticker == sequence.metadata.ticker_name:
                price = sequence.metadata.price_raw * (1 - self._additional_costs_perc)

                self._balance += price * stock.quantity
            else:
                new_inventory.append(stock)

        self._inventory = new_inventory

        return True

    def hold(self, sequence):
        return True

    def inventory_state(self, sequence):
        is_in_inventory = any(stock.ticker == sequence.metadata.ticker_name for stock in self._inventory)
        return int(is_in_inventory)

    @property
    def inventory(self):
        return self._inventory

    @property
    def balance(self):
        return self._balance
