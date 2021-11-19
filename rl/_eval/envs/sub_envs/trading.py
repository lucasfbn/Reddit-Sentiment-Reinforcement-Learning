from dataclasses import dataclass


@dataclass
class Stock:
    kind: str
    ticker: str
    price: float
    quantity: int


class TradingSimulator:
    START_BALANCE = 1000
    INVESTMENT_PER_TRADE = 70
    MAX_PRICE_PER_STOCK = None
    MAX_TRADES_PER_DAY = 3
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
    def _base_checks(operation):
        return operation.tradeable

    def buy(self, operation):
        if not self._base_checks(operation):
            return False

        if self._balance <= 0 or (self._balance - self.INVESTMENT_PER_TRADE) <= 0:
            return False

        if self.MAX_TRADES_PER_DAY is not None and self._n_trades >= self.MAX_TRADES_PER_DAY:
            return False

        if self.MAX_PRICE_PER_STOCK is not None and operation.price > self.MAX_PRICE_PER_STOCK:
            return False

        price = operation.price * (1 + self._additional_costs_perc)
        quantity = self.INVESTMENT_PER_TRADE // price

        self._inventory.append(Stock(kind="buy", ticker=operation.ticker, price=price, quantity=quantity))
        self._n_trades += 1
        self._balance -= price * quantity

        return True

    def sell(self, operation):

        if not self._base_checks(operation):
            return False

        new_inventory = []

        for stock in self._inventory:

            if stock.ticker == operation.ticker:
                price = operation.price * (1 - self._additional_costs_perc)

                self._balance += price * stock.quantity
            else:
                new_inventory.append(stock)

        self._inventory = new_inventory

        return True

    def hold(self, operation):
        return True

    @property
    def inventory(self):
        return self._inventory

    @property
    def balance(self):
        return self._balance
