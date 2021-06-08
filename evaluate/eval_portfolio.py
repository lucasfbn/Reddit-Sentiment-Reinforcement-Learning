import pickle as pkl
from dataclasses import dataclass, field

import mlflow
import pandas as pd
from tqdm import tqdm

from evaluate.actions import Buy, Sell
from utils.utils import log
from utils import mlflow_api


# log.setLevel("ERROR")


@dataclass
class EvaluatePortfolioInit:
    data: list
    initial_balance: int = 1000
    max_investment_per_trade: float = 0.07
    max_price_per_stock: int = 10
    max_trades_per_day: int = 3
    slippage: float = 0.007
    order_fee: float = 0.02
    partial_shares_possible: bool = False

    # Quantile of .85 means that we'll take the top 15%.
    quantiles_thresholds: dict = field(default_factory=dict)
    fixed_thresholds: bool = False
    live: bool = False

    balance = initial_balance
    profit = 1
    thresholds = None

    _inventory = []
    _extra_costs = 1 + slippage + order_fee

    _min_date = None
    _max_date = None
    _dates_trades_combination = None

    def get_result(self):
        return {"initial_balance": self.initial_balance,
                "max_investment_per_trade": self.max_investment_per_trade,
                "max_price_per_stock": self.max_price_per_stock,
                "max_trades_per_day": self.max_trades_per_day,
                "slippage": self.slippage,
                "partial_share_possible": self.partial_shares_possible,
                "quantiles_thresholds": self.quantiles_thresholds,
                "thresholds": self.thresholds,
                "order_fee": self.order_fee,
                "balance": self.balance,
                "profit": self.profit,
                "len_inventory": len(self._inventory),
                "index_performance": self.data[0]["index_comparison"]["perf"]}

    def log_result(self):
        mlflow.log_params(self.get_result())


class EvaluatePortfolio(EvaluatePortfolioInit):

    def _prepare_data(self):
        for grp in self.data:
            df = grp["metadata"]
            df["actions"] = df["actions"].replace({0: "hold", 1: "buy", 2: "sell"})
            grp["data"] = df

    def _check_live_settings(self):
        if self.live and not self.fixed_thresholds:
            raise ValueError("Ensure you are passing fixed (absolute, no quantiles) thresholds,"
                             " and set 'fixed_thresholds=True'")

    def initialize(self):
        self._check_live_settings()
        self._prepare_data()
        self._set_thresholds()

        if self._min_date is None and self._max_date is None:
            self._find_min_max_date()

        if self._dates_trades_combination is None:
            self._get_dates_trades_combination()

    def _set_thresholds(self):
        if self.fixed_thresholds:
            self.thresholds = {}
            for action, threshold in self.quantiles_thresholds.items():
                self.thresholds[action] = 0 if threshold is None else threshold
        else:
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

                if not (len(df_dict) == 1 or len(df_dict) == 0):
                    print()

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

            self._handle_sells(sells)
            self._handle_buys(potential_buys)

    def _handle_buys(self, potential_buys):
        Buy(portfolio=self, actions=potential_buys, live=self.live).execute()

    def _handle_sells(self, sells):
        Sell(portfolio=self, actions=sells, live=self.live).execute()

    def force_sell(self):
        log.warn("FORCING SELL OF REMAINING INVENTORY.")

        new_inventory = []
        for position in self._inventory:
            if not any(new_position["ticker"] == position["ticker"] for new_position in new_inventory):
                new_inventory.append({"ticker": position["ticker"], "tradeable": True})

        Sell(portfolio=self, actions=new_inventory, live=self.live, forced=True).execute()

    def save(self):
        mlflow_api.log_file(self, "state.pkl")
        log.info("Successfully saved state.")

    def load(self, path):
        with open(path, "rb") as f:
            state = pkl.load(f)
        self._inventory = state._inventory
        self.balance = state.balance
        self.profit = state.profit

        log.info("Successfully loaded state.")


class EvalLive(EvaluatePortfolio):

    def act(self):

        potential_buys = []
        sells = []

        for grp in self.data:
            trade = grp["data"].to_dict("records")
            assert len(trade) == 1
            trade = trade[0]
            trade["ticker"] = grp["ticker"]
            if trade["actions"] == "hold":
                continue
            elif trade["actions"] == "buy" and trade["tradeable"]:
                potential_buys.append(trade)
            elif trade["actions"] == "sell" and trade["tradeable"]:
                sells.append(trade)

        self._handle_sells(sells)
        self._handle_buys(potential_buys)


if __name__ == "__main__":
    import paths

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Evaluating")

    path = "C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/7/fb4e6edfba324080b424a40d85cdb48a/artifacts/eval_train.pkl"

    with mlflow.start_run():
        with open(path, "rb") as f:
            data = pkl.load(f)

        combination = {'max_trades_per_day': 3, 'max_price_per_stock': 20,
                       'max_investment_per_trade': 0.07,
                       'quantiles_thresholds': {'hold': None, 'buy': 0.9978524124622346, 'sell': None}}

        ep = EvaluatePortfolio(data=data, fixed_thresholds=True, **combination)
        ep.initialize()
        ep.act()
        ep.force_sell()
        print(ep.get_result())
        # cross_validate(data)
