import pickle as pkl

import mlflow
import pandas as pd

from eval.actions import Buy, Sell, ActionTracker
from eval.operation import Operation
from utils.mlflow_api import load_file, log_file
from utils.util_funcs import log

log.setLevel("DEBUG")


class Evaluate:

    def __init__(self,
                 ticker,
                 initial_balance=1000,
                 max_investment_per_trade=0.07,
                 max_price_per_stock=10,
                 max_trades_per_day=3,
                 slippage=0.007,
                 order_fee=0.02,
                 partial_shares_possible=False,
                 live=False):
        self.live = live
        self.partial_shares_possible = partial_shares_possible
        self.order_fee = order_fee
        self.slippage = slippage
        self.max_trades_per_day = max_trades_per_day
        self.max_price_per_stock = max_price_per_stock
        self.max_investment_per_trade = max_investment_per_trade
        self.initial_balance = initial_balance
        self.ticker = ticker

        self.balance = self.initial_balance
        self.profit = 1

        self.thresholds = {}
        self.inventory = []
        self._extra_costs = 1 + self.slippage + self.order_fee

        self.action_tracker = ActionTracker()

        self._min_date = None
        self._max_date = None
        self._dates_trades_combination = None

        self._sequence_attributes_df = None
        self._sequence_statistics = None

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
                "len_inventory": len(self.inventory)}

    def log_results(self):
        mlflow.log_params(self.get_result())

    def log_statistics(self):
        log_file(self._sequence_statistics, "eval_probability_stats.csv")
        log_file(self.action_tracker.get_actions(), "actions.csv")

        df = self.action_tracker.get_actions_stats()
        df[" "] = "|"

        results = pd.DataFrame([{"metric": key, "val": value} for key, value in self.get_result().items()])

        eval_stats = df.join(results, how="outer")
        log_file(eval_stats, "eval_stats.csv")

        log_file(self._sequence_attributes_df, "sequence_df.csv")

    def _rename_actions(self):
        map_ = {0: "hold", 1: "buy", 2: "sell"}
        for ticker in self.ticker:
            for sequence in ticker.sequences:
                sequence.action = map_[sequence.action]

    def _merge_sequence_attributes_to_df(self):
        dicts = []

        for ticker in self.ticker:
            for seq in ticker.sequences:
                seq_dict = dict(
                    ticker=ticker.name,
                    price=seq.price_raw,
                    date=seq.date,
                    tradeable=seq.tradeable,
                    action=seq.action,
                    action_probas=seq.action_probas,
                    hold=seq.action_probas["hold"],
                    buy=seq.action_probas["buy"],
                    sell=seq.action_probas["sell"],
                )
                dicts.append(seq_dict)

        self._sequence_attributes_df = pd.DataFrame(dicts)

    def _get_sequence_statistics(self):
        df = self._sequence_attributes_df.describe(percentiles=[0.25, 0.5, 0.75, 0.85, 0.9, 0.95])
        df["desc"] = df.index
        self._sequence_statistics = df

    def initialize(self):
        self._rename_actions()
        self._merge_sequence_attributes_to_df()
        self._get_sequence_statistics()
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
                same_action_df = self._sequence_attributes_df[self._sequence_attributes_df["action"] == action]

                # TODO Is this the correct behaviour?
                if same_action_df.empty:
                    self.thresholds[action] = 0.0
                    continue

                self.thresholds[action] = same_action_df[action].quantile(q=quantiles[action])

    def set_thresholds(self, thresholds):
        assert set(thresholds.keys()) == {"hold", "buy", "sell"}
        self.thresholds = thresholds

    def set_dates_trade_combination(self, dates_trades_combination):
        self._dates_trades_combination = dates_trades_combination

    def _get_dates_trades_combination(self):
        if self._dates_trades_combination is not None:
            return

        df = self._sequence_attributes_df

        # Generate a dict with the date range from min to max
        dates = pd.DataFrame(pd.date_range(df["date"].min().to_timestamp(), df["date"].max().to_timestamp()))
        dates[0] = dates[0].astype(str)
        dates = dates.set_index(0)
        dates["val"] = [[] for _ in range(len(dates))]
        dates = dates.T.to_dict("records")[0]

        # Group by date, convert grp rows to Operations and add them to the dates dict
        grps = df.groupby(["date"])

        for name, grp in grps:
            lst = []

            def to_operation(row):
                lst.append(
                    Operation(
                        ticker=row["ticker"],
                        price=row["price"],
                        date=name,
                        tradeable=row["tradeable"],
                        action=row["action"],
                        action_probas=row["action_probas"],
                    )
                )

            pd.DataFrame(grp).apply(to_operation, axis="columns")

            dates[name] = lst

        self._dates_trades_combination = dates

    def act(self):
        for day, trade_option in self._dates_trades_combination.items():

            log.debug(f"Processing day: {day}")

            potential_buys = []
            sells = []

            for operation in trade_option:
                if operation.action == "hold":
                    continue
                elif operation.action == "buy" and operation.tradeable:
                    potential_buys.append(operation)
                elif operation.action == "sell" and operation.tradeable:
                    sells.append(operation)

            self._handle_sells(sells)
            self._handle_buys(potential_buys)

            log.debug(f"Inventory len: {len(self.inventory)}")

    def _handle_buys(self, potential_buys):
        Buy(portfolio=self, actions=potential_buys, live=self.live).execute()

    def _handle_sells(self, sells):
        Sell(portfolio=self, actions=sells, live=self.live).execute()

    def force_sell(self):
        log.warn("FORCING SELL OF REMAINING INVENTORY.")

        new_inventory = []
        for position in self.inventory:
            if not any(new_position.ticker == position.ticker for new_position in new_inventory):
                position.tradeable = True
                new_inventory.append(position)

        Sell(portfolio=self, actions=new_inventory, live=self.live, forced=True).execute()

    def save(self):
        log_file(self, "state.pkl")
        log.info("Successfully saved state.")

    def load(self, path):
        with open(path, "rb") as f:
            state = pkl.load(f)
        self.inventory = state.inventory
        self.balance = state.balance
        self.profit = state.profit

        log.info("Successfully loaded state.")


class EvalLive(Evaluate):

    def initialize(self):
        self._rename_actions()

    def act(self):
        now = pd.Period.now("D")

        potential_buys = []
        sells = []

        for ticker in self.ticker:

            last_seq = ticker.sequences[len(ticker.sequences) - 1]
            if last_seq.date == now:

                operation = Operation(ticker=ticker.name,
                                      price=last_seq.price_raw,
                                      date=last_seq.date,
                                      tradeable=last_seq.tradeable,
                                      action=last_seq.action,
                                      action_probas=last_seq.action_probas)

                if operation.action == "hold":
                    continue
                elif operation.action == "buy" and operation.tradeable:
                    potential_buys.append(operation)
                elif operation.action == "sell" and operation.tradeable:
                    sells.append(operation)

        self._handle_sells(sells)
        self._handle_buys(potential_buys)


if __name__ == "__main__":
    import paths

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Tests")

    with mlflow.start_run():
        ticker = load_file(run_id="12829d4fd8fb408cbeee4d2e08f30c1f", experiment="N_Episodes_Impact_1", fn="eval.pkl")

        combination = {'max_trades_per_day': 3, 'max_price_per_stock': 20, 'max_investment_per_trade': 0.07}

        ep = Evaluate(ticker=ticker, **combination)
        ep.initialize()
        ep.set_quantile_thresholds({'hold': None, 'buy': 0.95, 'sell': None})
        ep.act()
        ep.force_sell()
        ep.log_results()
        ep.log_statistics()
        print(ep.get_result())
