import pandas as pd

from utils import log

log.setLevel("INFO")


class Action:

    def __init__(self, portfolio, actions, **kwargs):
        self.portfolio = portfolio
        self.p = portfolio  # Abbreviation for portfolio

        self.actions = actions
        self.actions_df = pd.DataFrame(actions)
        self.kwargs = kwargs

    def constraints(self):
        raise NotImplementedError

    def handle(self):
        raise NotImplementedError

    def callback(self, **kwargs):
        return True

    def execute(self):
        if not self.constraints():
            return
        self.handle()


class Buy(Action):

    def constraints(self):
        if len(self.actions_df) == 0:
            return False

        self.actions_df = self.actions_df[self.actions_df["tradeable"] == True]

        # TODO Quantile constraint

        if self.p.max_price_per_stock is not None:
            self.actions_df = self.actions_df[self.actions_df["price"] <= self.p.max_price_per_stock]

        if len(self.actions_df) == 0:
            return False
        return True

    def handle(self):
        self.actions_df = self.actions_df.sort_values(by=["actions_outputs"], ascending=True)
        self.actions_df = self.actions_df.to_dict("records")

        capital_per_trade = self.p.initial_balance * self.p.max_investment_per_trade
        i = 0

        for buy in self.actions_df:

            # Exit constraints should be above this statement
            if i == self.p.max_trades_per_day:
                break

            if not self.callback(buy=buy):
                continue
            else:
                i += 1

            buy["price"] *= self.p._extra_costs

            if self.p.partial_shares_possible:
                buyable_stocks = capital_per_trade / buy["price"]
            else:
                buyable_stocks = capital_per_trade // buy["price"]

            buy["quantity"] = buyable_stocks
            buy["total_buy_price"] = buyable_stocks * buy["price"]

            if self.p.balance - buy["total_buy_price"] <= 0:
                log.debug("Attempted BUY but balance is below or even to zero.")
                return

            old_depot = self.p.balance
            self.p.balance -= buy["total_buy_price"]

            self.p._inventory.append(buy)
            log.debug(f"BOUGHT. Ticker: {buy['ticker']}. "
                      f"Quantity: {buy['quantity']}. "
                      f"Total buy price: {buy['total_buy_price']}. "
                      f"Old depot: {old_depot}. "
                      f"New depot: {self.p.balance}")


class Sell(Action):

    def constraints(self):
        return True

    def handle(self):

        updated_inventory = []
        for position in self.p._inventory:

            delete = False

            for sell in self.actions:

                sell_ticker = sell["ticker"]
                position_ticker = position["ticker"]

                if sell_ticker == position_ticker and sell["tradeable"]:

                    bought_price = position["price"]
                    if "forced" in self.kwargs and self.kwargs["forced"]:
                        current_price = bought_price
                    else:
                        current_price = sell["price"]

                    profit_raw = current_price - bought_price
                    profit_perc = current_price / bought_price

                    if not self.p.sell_callback(sell, profit_perc):
                        continue

                    old_depot = self.p.balance
                    self.p.balance += current_price * position["quantity"]

                    self.p.profit = self.p.profit + \
                                    (self.p.profit * (self.p.max_investment_per_trade * (profit_perc - 1)))

                    log.debug(f"SOLD. Ticker: {position['ticker']}. "
                              f"Quantity: {position['quantity']}. "
                              f"Total buy price: {position['total_buy_price']}. "
                              f"Total sell price: {current_price * position['quantity']} "
                              f"Relative profit: {profit_perc} "
                              f"Old depot: {old_depot}. "
                              f"New depot: {self.p.balance}")

                    delete = True

            if not delete:
                updated_inventory.append(position)

        self.p._inventory = updated_inventory
