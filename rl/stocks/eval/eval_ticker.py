from tqdm import tqdm

from rl.utils.predict_proba import predict_proba
import pandas as pd


class Tracker:

    def __init__(self, ticker_name):
        self.ticker_name = ticker_name

        self._actions = []
        self._prices = []
        self._rewards = []
        self._timesteps = 0
        self._dates = []
        self._metadata = []

    def add(self, action, price, reward, date):
        self._actions.append(action)
        self._prices.append(price)
        self._rewards.append(reward)
        self._timesteps += 1
        self._dates.append(str(date))

    def add_metadata(self, info: dict):
        self._metadata.append(info)

    def make_dict(self):
        d = {
            "metadata": {
                "ticker": self.ticker_name,
                "profit": sum(self._rewards),
            },
            "points": {
                "actions": self._actions,
                "prices": self._prices,
                "timesteps": list(range(self._timesteps)),
                "rewards": self._rewards,
                "dates": self._dates
            }
        }

        for metadata in self._metadata:
            d["metadata"] = d["metadata"] | metadata

        return d


class AggStats:

    def __init__(self, tracker_df):
        self.tracker_df = tracker_df

    @staticmethod
    def q(n):
        def percentile_(x):
            return x.quantile(n)

        percentile_.__name__ = str(n)
        return percentile_

    @staticmethod
    def pos(series):
        return sum(series > 0) / len(series)

    @staticmethod
    def even(series):
        return sum(series == 0) / len(series)

    @staticmethod
    def neg(series):
        return sum(series < 0) / len(series)

    @staticmethod
    def mean_neg(series):
        return series[series < 0].mean()

    @staticmethod
    def mean_pos(series):
        return series[series > 0].mean()

    def agg(self):
        df = self.tracker_df.select_dtypes(include="number")
        agg = df.agg(["count", "mean", "min", "max", self.q(0.25), self.q(0.50), self.q(0.75),
                      self.q(0.9), self.q(0.95), self.pos, self.even, self.neg, self.mean_neg, self.mean_pos])
        agg = agg.round(4)
        agg["func"] = agg.index

        cols = agg.columns.tolist()
        agg = agg[cols[-1:] + cols[:-1]]
        return agg


class Eval:

    def __init__(self, ticker, model, training_env_cls, trading_env_cls):
        self.trading_env_cls = trading_env_cls
        self.training_env_cls = training_env_cls
        self.model = model
        self.ticker = ticker

        self._all_tracker = []

    def _eval_sequence(self, sequence, trading_env):

        state = self.training_env_cls.state_handler.forward(sequence,
                                                            trading_env.inventory_state())
        action, _ = predict_proba(self.model, state)

        if action == 0:
            reward = trading_env.hold(sequence.price)
        elif action == 1:
            reward = trading_env.buy(sequence.price)
        elif action == 2:
            reward = trading_env.sell(sequence.price)
        else:
            raise ValueError("Invalid action.")

        return action, reward

    def _eval_sequences(self, sequences, tracker):

        trading_env = self.trading_env_cls()

        for seq in sequences:
            state = self.training_env_cls.state_handler.forward(seq,
                                                                trading_env.inventory_state())
            action, _ = predict_proba(self.model, state)

            if action == 0:
                reward = trading_env.hold(seq.price)
            elif action == 1:
                reward = trading_env.buy(seq.price)
            elif action == 2:
                reward = trading_env.sell(seq.price)
            else:
                raise ValueError("Invalid action.")

            tracker.add(action, seq.price, reward, seq.date)

        tracker.add_metadata({"open_positions": len(trading_env.inventory)})
        tracker.add_metadata({"min_date": str(sequences[0].date)})
        tracker.add_metadata({"max_date": str(sequences[len(sequences) - 1].date)})

    def eval_ticker(self):
        for ticker in tqdm(self.ticker):
            tracker = Tracker(ticker.name)

            self._eval_sequences(ticker.sequences, tracker)

            self._all_tracker.append(tracker)

    @property
    def all_tracker(self):
        return self._all_tracker

    @property
    def all_tracker_dict(self):
        return [tracker.make_dict() for tracker in self._all_tracker]

    @property
    def agg_metadata_df(self):
        return pd.DataFrame([ticker["metadata"] for ticker in self.all_tracker_dict])

    @property
    def agg_metadata_stats(self):
        agg_stats = AggStats(self.agg_metadata_df)
        return agg_stats.agg()

    @property
    def agg_metadata_stats_flat(self):
        return pd.json_normalize(self.agg_metadata_stats.drop(columns=["func"]).to_dict(),
                                 sep="_").to_dict(orient="records")[0]
