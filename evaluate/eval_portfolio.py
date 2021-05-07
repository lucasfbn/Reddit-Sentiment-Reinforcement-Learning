import pickle as pkl

import mlflow
import pandas as pd
from tqdm import tqdm

from evaluate.actions import Buy, Sell
from utils import log

log.setLevel("ERROR")


class EvaluatePortfolio:

    def __init__(self, eval_data,
                 initial_balance=1000,
                 max_investment_per_trade=0.05,
                 max_price_per_stock=25,
                 max_trades_per_day=5,
                 slippage=0.007,
                 order_fee=0.02,
                 partial_shares_possible=True,

                 # Quantile of .85 means that we'll take the top 15%.
                 quantiles_thresholds={"hold": None, "buy": 0.5, "sell": None}
                 ):

        self.data = eval_data

        self.initial_balance = initial_balance
        self.max_investment_per_trade = max_investment_per_trade
        self.max_price_per_stock = max_price_per_stock
        self.max_trades_per_day = max_trades_per_day
        self.quantiles_thresholds = quantiles_thresholds
        self.thresholds = None

        self.slippage = slippage
        self.order_fee = order_fee
        self._extra_costs = 1 + slippage + order_fee

        self.partial_shares_possible = partial_shares_possible

        self._min_date = None
        self._max_date = None

        self.balance = initial_balance
        self.profit = 1

        self._inventory = []

        self._dates_trades_combination = None

    def initialize(self):
        for grp in self.data:
            df = grp["metadata"]
            df["actions"] = df["actions"].replace({0: "hold", 1: "buy", 2: "sell"})
            grp["data"] = df

        self._find_min_max_date()

        if self._dates_trades_combination is None:
            self._get_dates_trades_combination()

        self.thresholds = self._calculate_thresholds(self.quantiles_thresholds)

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
        # TODO Improve performance (flatten)
        log.info("Retrieving date/trades combinations...")
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

        self._dates_trades_combination = dates_trades_combinations

    def _calculate_thresholds(self, quantiles):
        thresholds = {}

        for action, quantile in quantiles.items():
            if quantile is None:
                # Since the actions are a probability distribution between 3 classes,
                # setting the quantile (which we will later filter for) to 0 will guarantee that we don't filter
                # anything (as every probability is at least >= 0)
                thresholds[action] = 0
            else:
                same_action_value = []
                for grp in self.data:
                    df = grp["data"]
                    df = df[df["actions"] == action]
                    same_action_value.extend(df[action + "_probability"].tolist())
                same_action_value = pd.Series(same_action_value)
                thresholds[action] = same_action_value.quantile(q=quantiles[action])
        return thresholds

    def act(self):
        inventory = []

        for day, trade_option in self._dates_trades_combination.items():

            potential_buys = []
            sells = []

            for trade in trade_option:
                if trade["actions"] == "hold":
                    continue
                elif trade["actions"] == "buy" and trade["tradeable"]:
                    potential_buys.append(trade)
                elif trade["actions"] == "sell" and trade["tradeable"]:
                    sells.append(trade)

            df = pd.DataFrame(sells)
            a = df.to_dict("records")

            self._handle_sells(sells)
            self._handle_buys(potential_buys)

    def _handle_buys(self, potential_buys):
        Buy(portfolio=self, actions=potential_buys).execute()

    def _handle_sells(self, sells):
        Sell(portfolio=self, actions=sells).execute()

    def force_sell(self):
        log.warn("FORCING SELL OF REMAINING INVENTORY.")

        new_inventory = []
        for position in self._inventory:
            if not any(new_position["ticker"] == position["ticker"] for new_position in new_inventory):
                new_inventory.append({"ticker": position["ticker"], "tradeable": True})

        Sell(portfolio=self, actions=new_inventory, forced=True).execute()

    def log_state(self):

        mlflow.log_params({"initial_balance": self.initial_balance,
                           "max_investment_per_trade": self.max_investment_per_trade,
                           "max_price_per_stock": self.max_price_per_stock,
                           "max_trades_per_day": self.max_trades_per_day, "slippage": self.slippage,
                           "partial_share_possible": self.partial_shares_possible,
                           "quantiles_thresholds": self.quantiles_thresholds,
                           "thresholds": self.thresholds,
                           "order_fee": self.order_fee, "profit": self.profit, "balance": self.balance})


if __name__ == "__main__":
    import paths

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Evaluating")

    path = "C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/5/" \
           "472e633695ce4beab58634b5e73d10c2/artifacts/eval_test_0.pkl"

    with mlflow.start_run():
        with open(path, "rb") as f:
            data = pkl.load(f)
        data = data[:10]

        ep = EvaluatePortfolio(eval_data=data)
        ep.initialize()
        ep.act()
        ep.force_sell()
        ep.log_state()
        print(ep.thresholds)

        print(ep.profit)
        print(ep.balance)

        # cross_validate(data)
