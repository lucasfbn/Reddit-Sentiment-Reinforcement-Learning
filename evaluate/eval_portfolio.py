import datetime
import pickle as pkl
import warnings

import pandas as pd
from tqdm import tqdm

from utils import log
from evaluate.actions import Buy, Sell

log.setLevel("DEBUG")


class EvaluatePortfolio:

    def __init__(self, eval_data,
                 initial_balance=1000,
                 max_investment_per_trade=0.05,
                 max_price_per_stock=25,
                 max_trades_per_day=5,
                 slippage=0.007,
                 order_fee=0.02,
                 partial_shares_possible=True,
                 quantiles={"buy": None, "hold": None, "sell": None}
                 ):

        self.data = eval_data
        self.prepare()

        self.initial_balance = initial_balance
        self.max_investment_per_trade = max_investment_per_trade
        self.max_price_per_stock = max_price_per_stock
        self.max_trades_per_day = max_trades_per_day

        self.quantiles = quantiles

        # self.action_outputs = self._action_outputs_df()
        # self.max_buy_output_quantile = max_buy_output_quantile
        #
        # if max_buy_output is not None:
        #     warnings.warn("max_buy_output is != None. max_buy_output_quantile will be ignored.")
        #     self.max_buy_output = max_buy_output
        # elif len(self.action_outputs) != 0:
        #     self.max_buy_output = float(self.action_outputs.quantile(max_buy_output_quantile))
        # else:
        self.max_buy_output = 0

        self.slippage = slippage
        self.order_fee = order_fee
        self._extra_costs = 1 + slippage + order_fee

        self.partial_shares_possible = partial_shares_possible

        self._min_date = None
        self._max_date = None

        self.balance = initial_balance
        self.profit = 1

        self._inventory = []

    def prepare(self):
        for grp in self.data:
            df = grp["metadata"]
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

        for date in tqdm(dates):
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

    # def _calc_quantiles(self, quantiles):

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

            # self._handle_sells(pd.DataFrame(sells))
            # self._handle_buys(pd.DataFrame(potential_buys))

            log.info(len(potential_buys))
            log.info(len(sells))

            self._handle_sells(sells)
            self._handle_buys(potential_buys)

    def _handle_buys(self, potential_buys):
        Buy(portfolio=self, actions=potential_buys).execute()

    def _handle_sells(self, sells, forced=False):
        Sell(portfolio=self, actions=sells).execute()

    def force_sell(self):
        warnings.warn("FORCING SELL OF REMAINING INVENTORY.")

        new_inventory = []
        for position in self._inventory:
            if not any(new_position["ticker"] == position["ticker"] for new_position in new_inventory):
                new_inventory.append({"ticker": position["ticker"], "tradeable": True})

        Sell(portfolio=self, actions=new_inventory, forced=True).execute()

    def _action_outputs_df(self):

        action_outputs = []

        for grp in self.data:
            df = grp["data"]
            df = df[df["actions"] == "buy"]
            df = self._buy_constraints(df)
            action_outputs += df["actions_outputs"].values.tolist()

        return pd.DataFrame(action_outputs)

    def buy_callback(self, buy):
        # Used to implement custom callbacks when buying. While evaluating we do not need such callbacks.
        return True

    def sell_callback(self, sell, profit_perc):
        # Used to implement custom callbacks when selling. While evaluating we do not need such callbacks.
        return True

    def report(self):

        mlflow.log_params({"initial_balance": [self.initial_balance],
                           "max_investment_per_trade": [self.max_investment_per_trade],
                           "max_price_per_stock": [self.max_price_per_stock],
                           "max_buy_output_quantile": [self.max_buy_output_quantile],
                           "max_buy_output": [self.max_buy_output],
                           "max_trades_per_day": [self.max_trades_per_day], "slippage": [self.slippage],
                           "partial_share_possible": [self.partial_shares_possible],
                           "order_fee": [self.order_fee], "profit": [self.profit], "balance": [self.balance],
                           "time": datetime.datetime.now().strftime("%Hh%Mm %d_%m-%y")})


if __name__ == "__main__":
    import paths
    import mlflow

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Evaluating")

    with mlflow.start_run():
        with open(
                "C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/5/79b744d7a4b94b15b21eddf703927b9f/artifacts/eval_test_0.pkl",
                "rb") as f:
            data = pkl.load(f)
        for d in data:
            d["metadata"]["actions_outputs"] = 1
        ep = EvaluatePortfolio(eval_data=data)
        # print(ep.action_outputs.describe())

        ep.act()
        ep.force_sell()

        print(ep.profit)
        print(ep.balance)
