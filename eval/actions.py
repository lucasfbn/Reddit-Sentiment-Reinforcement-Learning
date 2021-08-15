import mlflow

import pandas as pd

from utils.util_funcs import log

log.setLevel("INFO")


class Action:
    action_name = None
    _actions_lst = []

    def __init__(self, portfolio, actions, live, **kwargs):
        self.portfolio = portfolio
        self.p = portfolio  # Abbreviation for portfolio

        self.actions = actions

        self.live = live

        self.kwargs = kwargs

    def base_constraints(self):
        if len(self.actions) == 0:
            return False

        # Keep only tradeable sequences
        self.actions = [action for action in self.actions if action.tradeable]

        # self.actions_df = self.actions_df[self.actions_df["tradeable"] == True]

        old_len = len(self.actions)

        # Probability must be above threshold
        self.actions = [action for action in self.actions if
                        action.action_probas[self.action_name] >= self.p.thresholds[self.action_name]]

        if self.p.thresholds[self.action_name] == 0:
            assert old_len == len(self.actions)
        return True

    def constraints(self):
        raise NotImplementedError

    def handle(self):
        raise NotImplementedError

    def callback(self, action):
        if not self.live:
            return True
        else:
            decision = input(f"Attempting to {self.action_name} {action.ticker} "
                             f"@ {action.price}. Execute {self.action_name}? (y/n)")
            if decision == "y":
                return True
            return False

    def log(self):
        mlflow.log_metric("balance", self.p.balance)
        mlflow.log_metric("Inventory Length", len(self.p._inventory))

    def execute(self):
        if not self.constraints():
            return
        self.handle()
        # self.log()

    @staticmethod
    def get_actions():
        return pd.DataFrame(Action._actions_lst)

    @staticmethod
    def get_actions_stats():
        return Action.get_actions().describe(percentiles=[0.25, 0.5, 0.75, 0.8, 0.9, 0.95])


class Buy(Action):
    action_name = "buy"

    def constraints(self):
        if not self.base_constraints():
            return False

        # Only keep those where price below max price
        if self.p.max_price_per_stock is not None:
            self.actions = [action for action in self.actions if action.price <= self.p.max_price_per_stock]

        if len(self.actions) == 0:
            return False
        return True

    def handle(self):
        self.actions.sort(key=lambda x: x.price, reverse=False)

        capital_per_trade = self.p.initial_balance * self.p.max_investment_per_trade
        i = 0

        for buy in self.actions:

            # Exit constraints should be above this statement
            if i == self.p.max_trades_per_day:
                break

            if not self.callback(action=buy):
                continue
            else:
                i += 1

            price = buy.price * self.p._extra_costs

            if self.p.partial_shares_possible:
                buyable_stocks = capital_per_trade / price
            else:
                buyable_stocks = capital_per_trade // price

            quantity = buyable_stocks
            total_buy_price = buyable_stocks * price

            if self.p.balance - total_buy_price <= 0:
                log.debug("Attempted BUY but balance is below or even to zero.")
                return

            buy.save_buy(price, quantity, total_buy_price)

            old_depot = self.p.balance
            self.p.balance -= total_buy_price
            self.p._inventory.append(buy)

            Action._actions_lst.append(
                dict(
                    date=str(buy.date),
                    action="buy",
                    ticker=buy.ticker,
                    price=price,
                    quantity=buy.quantity,
                    total=buy.total_buy_price,
                    old_balance=old_depot,
                    new_balance=self.p.balance
                )
            )

            log.debug(f"BOUGHT. Ticker: {buy.ticker}. "
                      f"Quantity: {buy.quantity}. "
                      f"Total buy price: {buy.total_buy_price}. "
                      f"Old depot: {old_depot}. "
                      f"New depot: {self.p.balance}")


class Sell(Action):
    action_name = "sell"

    def constraints(self):
        if "forced" in self.kwargs and self.kwargs["forced"]:
            return True

        if not self.base_constraints():
            return False

        return True

    def handle(self):

        updated_inventory = []
        for position in self.p._inventory:

            delete = False

            for sell in self.actions:

                sell_ticker = sell.ticker
                position_ticker = position.ticker

                if sell_ticker == position_ticker and sell.tradeable:

                    forced = False

                    bought_price = position.price_bought
                    if "forced" in self.kwargs and self.kwargs["forced"]:
                        current_price = bought_price
                        forced = True
                    else:
                        current_price = sell.price

                    profit_raw = current_price - bought_price
                    profit_perc = current_price / bought_price

                    if not self.callback(action=sell):
                        continue

                    old_depot = self.p.balance
                    self.p.balance += current_price * position.quantity
                    self.p.profit = self.p.profit + \
                                    (self.p.profit * (self.p.max_investment_per_trade * (profit_perc - 1)))

                    Action._actions_lst.append(
                        dict(
                            date=str(position.date),
                            action="sell",
                            ticker=position.ticker,
                            price=current_price,
                            quantity=position.quantity,
                            total=current_price * position.quantity,
                            old_balance=old_depot,
                            new_balance=self.p.balance,
                            profit=profit_perc,
                            profit_raw=profit_raw,
                            forced=forced
                        )
                    )

                    log.debug(f"SOLD. Ticker: {position.ticker}. "
                              f"Quantity: {position.quantity}. "
                              f"Total buy price: {position.total_buy_price}. "
                              f"Total sell price: {current_price * position.quantity} "
                              f"Relative profit: {profit_perc} "
                              f"Old depot: {old_depot}. "
                              f"New depot: {self.p.balance}")

                    delete = True

            if not delete:
                updated_inventory.append(position)

        self.p._inventory = updated_inventory
