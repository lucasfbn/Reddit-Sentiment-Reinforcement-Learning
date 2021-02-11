import time
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

verbose = False

if verbose:
    def vprint(msg):
        print(msg)
else:
    def vprint(msg):
        pass


class EvaluatePortfolio:

    def __init__(self, data,
                 initial_balance=1000000,
                 max_investment_per_trade=0.02,
                 max_price_per_stock=9999999,
                 min_buy_output=0.9,
                 max_trades_per_day=25,
                 slippage=0.007,
                 order_fee=0.02):

        self.data = data

        self.initial_balance = initial_balance
        self.max_investment_per_trade = max_investment_per_trade
        self.max_price_per_stock = max_price_per_stock
        self.min_buy_output = min_buy_output
        self.max_trades_per_day = max_trades_per_day

        self._extra_costs = 1 + slippage + order_fee

        self.prepare()

        self._min_date = None
        self._max_date = None

        self.balance = initial_balance
        self.profit = 1

        self._inventory = []
        self._log = []

    def prepare(self):
        for grp in self.data:
            df = grp["data"]
            df = df.drop(df.tail(1).index)  # since we do not have an action for the last entry
            df = df[["Close", "actions", "actions_outputs", "tradeable", "date"]]
            df["actions"] = df["actions"].replace({0: "hold", 1: "buy", 2: "sell"})
            grp["data"] = df

    def _find_min_max_date(self):
        min_date = None
        max_date = None

        for i, grp in enumerate(self.data):
            df = grp["data"]

            if i == 0:
                min_date = df["date"].min()
                max_date = df["date"].max()

            if df["date"].min() < min_date:
                min_date = df["date"].min()

            if df["date"].max() > max_date:
                max_date = df["date"].max()

        self._min_date = min_date
        self._max_date = max_date

    def _get_dates_trades_combination(self):
        dates = pd.date_range(self._min_date, self._max_date)

        dates_trades_combinations = {}

        for date in dates:
            dates_trades_combinations[date.strftime("%d-%m-%Y")] = []
            for grp in self.data:
                df = grp["data"]
                df = df[(df["date"] - date).dt.days == 0]
                df_dict = df.to_dict("records")
                assert len(df_dict) == 1 or len(df_dict) == 0

                if df_dict:
                    df_dict = df_dict[0]
                    df_dict["ticker"] = grp["ticker"]
                    dates_trades_combinations[date.strftime("%d-%m-%Y")].append(df_dict)

        return dates_trades_combinations

    def act(self):
        self._find_min_max_date()

        dates_trades_combinations = self._get_dates_trades_combination()

        inventory = []

        for day, trade_option in dates_trades_combinations.items():

            potential_buys = []
            sells = []

            for trade in trade_option:
                if trade["actions"] == "hold":
                    continue
                elif trade["actions"] == "buy" and trade["tradeable"]:
                    potential_buys.append(trade)
                elif trade["actions"] == "sell" and trade["tradeable"]:
                    sells.append(trade)

            self._handle_buys(potential_buys)
            self._handle_sells(sells)

    def _handle_buys(self, potential_buys):

        if len(potential_buys) == 0:
            return

        df = pd.DataFrame(potential_buys)
        df = df[df["tradeable"] == True]  # Only keep tradeable ticker

        df = df[df["actions_outputs"] >= self.min_buy_output]
        df = df[df["Close"] <= self.max_price_per_stock]

        if len(df) == 0:
            return []

        df = df.sort_values(by=["actions_outputs"], ascending=False)

        reserve = df.to_dict("records")
        df = df.head(self.max_trades_per_day)
        buys = df.to_dict("records")

        self._execute_buy(buys)

    def _execute_buy(self, buys):

        capital_per_trade = self.initial_balance * self.max_investment_per_trade

        for buy in buys:

            buy["Close"] *= self._extra_costs
            buyable_stocks = capital_per_trade / buy["Close"]

            buy["quantity"] = buyable_stocks
            buy["total_buy_price"] = buyable_stocks * buy["Close"]

            if self.balance - buy["total_buy_price"] <= 0:
                vprint("Attempted BUY but balance is below or even to zero.")
                return

            old_depot = self.balance
            self.balance -= buy["total_buy_price"]

            self._inventory.append(buy)
            self._log.append({"buy": buy, "old_depot": old_depot, "new_depot": self.balance})

            vprint(f"BOUGHT. Ticker: {buy['ticker']}. "
                   f"Quantity: {buy['quantity']}. "
                   f"Total buy price: {buy['total_buy_price']}. "
                   f"Old depot: {old_depot}. "
                   f"New depot: {self.balance}")

    def _handle_sells(self, sells, forced=False):

        updated_inventory = []
        for position in self._inventory:

            delete = False

            for sell in sells:

                sell_ticker = sell["ticker"]
                position_ticker = position["ticker"]

                if sell_ticker == position_ticker and sell["tradeable"]:

                    bought_price = position["Close"]
                    if forced:
                        current_price = bought_price
                    else:
                        current_price = sell["Close"]

                    old_depot = self.balance
                    self.balance += current_price * position["quantity"]

                    profit_raw = current_price - bought_price
                    profit_perc = current_price / bought_price

                    self.profit = self.profit + (self.profit * (self.max_investment_per_trade * (profit_perc - 1)))

                    self._log.append(
                        {"sell": position,
                         "profit_raw": profit_raw, "profit_perc": profit_perc,
                         "old_depot": old_depot, "new_depot": self.balance})

                    vprint(f"SOLD. Ticker: {position['ticker']}. "
                           f"Quantity: {position['quantity']}. "
                           f"Total buy price: {position['total_buy_price']}. "
                           f"Total sell price: {current_price * position['quantity']} "
                           f"Relative profit: {profit_perc} "
                           f"Old depot: {old_depot}. "
                           f"New depot: {self.balance}"
                           )

                    delete = True

            if not delete:
                updated_inventory.append(position)

        self._inventory = updated_inventory

    # def plot(self):

    def force_sell(self):
        warnings.warn("FORCING SELL OF REMAINING INVENTORY.")

        new_inventory = []
        for position in self._inventory:
            if not any(new_position["ticker"] == position["ticker"] for new_position in new_inventory):
                new_inventory.append({"ticker": position["ticker"], "tradeable": True})

        self._handle_sells(new_inventory, forced=True)

    def get_log(self):
        return pd.DataFrame(self._log)


import pickle as pkl
import paths

with open(paths.models_path / "18_44---08_02-21.mdl" / "eval_test.pkl", "rb") as f:
    data = pkl.load(f)

# data = data[:10]

ep = EvaluatePortfolio(data)
ep.act()
ep.force_sell()

print()
print(ep.profit)
print(ep.balance)
