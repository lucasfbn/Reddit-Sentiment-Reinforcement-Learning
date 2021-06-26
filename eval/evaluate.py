import pickle as pkl
from dataclasses import dataclass

import mlflow
import pandas as pd
from tqdm import tqdm

from eval.actions import Buy, Sell
from utils import mlflow_api
from utils.util_funcs import log

log.setLevel("INFO")


@dataclass
class Operation:
    ticker: str
    price: float
    date: pd.Period
    tradeable: bool
    action: str
    action_probas: dict

    price_bought: float = None
    quantity: int = None
    total_buy_price: float = None

    def save_buy(self, price, quantity, total_buy_price):
        self.price_bought = price
        self.quantity = quantity
        self.total_buy_price = total_buy_price


@dataclass
class EvaluateInit:
    ticker: list
    initial_balance: int = 1000
    max_investment_per_trade: float = 0.07
    max_price_per_stock: int = 10
    max_trades_per_day: int = 3
    slippage: float = 0.007
    order_fee: float = 0.02
    partial_shares_possible: bool = False

    live: bool = False

    balance = None
    profit = 1

    thresholds = {}
    _inventory = []
    _extra_costs = None

    _sequence_attributes_df = None

    _min_date = None
    _max_date = None
    _dates_trades_combination = None

    def __post_init__(self):
        self.balance = self.initial_balance
        self._extra_costs = 1 + self.slippage + self.order_fee

    def get_result(self):
        return {"initial_balance": self.initial_balance,
                "max_investment_per_trade": self.max_investment_per_trade,
                "max_price_per_stock": self.max_price_per_stock,
                "max_trades_per_day": self.max_trades_per_day,
                "slippage": self.slippage,
                "thresholds": self.thresholds,
                "order_fee": self.order_fee,
                "balance": self.balance,
                "profit": self.profit,
                "len_inventory": len(self._inventory)}

    def log_result(self):
        mlflow.log_params(self.get_result())


class Evaluate(EvaluateInit):

    def _rename_actions(self):
        map_ = {0: "hold", 1: "buy", 2: "sell"}
        for ticker in self.ticker:
            for sequence in ticker.sequences:
                sequence.action = map_[sequence.action]

    def _merge_sequence_attributes_to_df(self):
        dates = []
        actions = []
        action_probas = []

        for ticker in self.ticker:
            for seq in ticker.sequences:
                dates.append(seq.date)
                actions.append(seq.action)
                action_probas.append(seq.action_probas)

        df = pd.DataFrame(action_probas)
        df["dates"] = dates
        df["actions"] = actions
        self._sequence_attributes_df = df

    def initialize(self):
        self._rename_actions()
        self._merge_sequence_attributes_to_df()
        self._find_min_max_date()
        self._get_dates_trades_combination()

    def set_quantile_thresholds(self, quantiles):
        assert set(quantiles.keys()) == {"hold", "buy", "sell"}

        for action, quantile in quantiles.items():
            if quantile is None:
                # Since the actions are a probability distribution between 3 classes,
                # setting the quantile (which we will later filter for) to 0 will guarantee that we don't filter
                # anything (as every probability is at least >= 0)
                self.thresholds[action] = 0.0
            else:
                same_action_df = self._sequence_attributes_df[self._sequence_attributes_df["actions"] == action]

                # TODO Is this the correct behaviour?
                if same_action_df.empty:
                    self.thresholds[action] = 0.0
                    continue

                action_probas = same_action_df[action]
                self.thresholds[action] = action_probas.quantile(q=quantiles[action])

    def set_thresholds(self, thresholds):
        assert set(thresholds.keys()) == {"hold", "buy", "sell"}
        self.thresholds = thresholds

    def set_min_max_date(self, min_date, max_date):
        self._min_date, self._max_date = min_date, max_date

    def set_dates_trade_combination(self, dates_trades_combination):
        self._dates_trades_combination = dates_trades_combination

    def _find_min_max_date(self):
        if self._min_date is not None and self._max_date is not None:
            return

        self._min_date = self._sequence_attributes_df["dates"].min()
        self._max_date = self._sequence_attributes_df["dates"].max()

    def _get_dates_trades_combination(self):
        if self._dates_trades_combination is not None:
            return

        log.info("Retrieving date/trades combinations...")
        dates = pd.date_range(self._min_date.to_timestamp(), self._max_date.to_timestamp())

        dates_trades_combinations = {}

        for date in tqdm(dates):
            dates_trades_combinations[date.strftime("%d-%m-%Y")] = []
            for ticker in self.ticker:

                for seq in ticker.sequences:

                    if (seq.date.to_timestamp() - date).days == 0:
                        dates_trades_combinations[date.strftime("%d-%m-%Y")].append(Operation(ticker=ticker.name,
                                                                                              price=seq.price_raw,
                                                                                              date=date,
                                                                                              tradeable=seq.tradeable,
                                                                                              action=seq.action,
                                                                                              action_probas=seq.action_probas))

        self._dates_trades_combination = dates_trades_combinations

    def act(self):
        for day, trade_option in self._dates_trades_combination.items():

            potential_buys = []
            sells = []

            for seq in trade_option:
                if seq.action == "hold":
                    continue
                elif seq.action == "buy" and seq.tradeable:
                    potential_buys.append(seq)
                elif seq.action == "sell" and seq.tradeable:
                    sells.append(seq)

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


class EvalLive(Evaluate):

    def act(self):

        potential_buys = []
        sells = []

        for grp in self.ticker:
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

    ticker = mlflow_api.load_file("6bc3213bdfa84c7f862302b13ef2a21b", "eval.pkl", experiment="Tests")

    with mlflow.start_run():
        combination = {'max_trades_per_day': 3, 'max_price_per_stock': 20, 'max_investment_per_trade': 0.07}

        ep = Evaluate(ticker=ticker, **combination)
        ep.set_thresholds({'hold': None, 'buy': None, 'sell': None})
        ep.initialize()
        ep.act()
        ep.force_sell()
        print(ep.get_result())
