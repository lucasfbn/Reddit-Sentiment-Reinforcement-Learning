from tqdm import tqdm

from rl.utils.predict_proba import predict_proba
import pandas as pd


class Eval:

    def __init__(self, ticker, model, training_env_cls, trading_env_cls):
        self.trading_env_cls = trading_env_cls
        self.training_env_cls = training_env_cls
        self.model = model
        self.ticker = ticker

        self._all_tracker = []

    def _eval_sequences(self, sequences):

        trading_env = self.trading_env_cls()

        for seq in sequences:
            state = self.training_env_cls.state_handler.forward(seq,
                                                                trading_env.inventory_state())
            action, proba = predict_proba(self.model, state)

            if action == 0:
                reward = trading_env.hold(seq.metadata.price)
            elif action == 1:
                reward = trading_env.buy(seq.metadata.price)
            elif action == 2:
                reward = trading_env.sell(seq.metadata.price)
            else:
                raise ValueError("Invalid action.")

            seq.evl.action = action
            seq.evl.reward = reward
            seq.evl.probas = proba
            seq.evl.open_positions = len(trading_env.inventory)
            seq.evl.split_probas()

        return len(trading_env.inventory)

    def eval_ticker(self):
        for ticker in tqdm(self.ticker):
            ticker.evl.open_positions = self._eval_sequences(ticker.sequences)

        _ = [(ticker.drop_sequence_data(), ticker.aggregate_rewards()) for ticker in self.ticker]
        return self.ticker

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
