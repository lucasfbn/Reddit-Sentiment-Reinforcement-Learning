from tqdm import tqdm

from rl.portfolio.eval.envs.sub_envs.trading import TradingSimulator
from rl.portfolio.eval.envs.tracker.track import Tracker, EnvStateTracker
from rl.utils.predict_proba import predict_proba
from rl.portfolio.eval.envs.utils.utils import *


class EvalEnv:

    def __init__(self, ticker, pre_processor, training_env, model):
        self.ticker = ticker
        self.pre_processor = pre_processor
        self.training_env = training_env
        self.model = model

        self._trading_env = TradingSimulator()

        self.detail_tracker = Tracker(self._trading_env)
        self.overall_tracker = EnvStateTracker(self._trading_env)

    def run_pre_processor(self):
        self.ticker = self.pre_processor.run(self.ticker)

    def eval_loop(self):
        ticker_df = ticker_list_to_df(self.ticker)
        ordered_day_wise = order_day_wise(self.ticker, ticker_df)

        for day, operations in tqdm(ordered_day_wise.items(), desc="Processing day"):

            action_pairs = []

            for operation in operations:
                state = self.training_env.state_handler.forward(operation.sequence,
                                                                self._trading_env.inventory_state(operation))

                action_direct, _ = self.model.predict(state, deterministic=True)
                action_own, proba = predict_proba(model=self.model, state=state)

                assert int(action_direct) == action_own

                action = action_own
                action_pairs.append(dict(action=action, proba=proba, operation=operation))

            holds = [pair for pair in action_pairs if pair["action"] == 0]
            buys = [pair for pair in action_pairs if pair["action"] == 1]
            sells = [pair for pair in action_pairs if pair["action"] == 2]

            # Sort buys by probability in descending order
            buys = sorted(buys, key=lambda tup: tup["proba"][1], reverse=True)

            # Execute sells first
            for sell in sells:
                success = self._trading_env.sell(sell["operation"])
                self.detail_tracker.track(day, success, sell)

            for hold in holds:
                success = self._trading_env.hold(hold["operation"])
                self.detail_tracker.track(day, success, hold)

            for buy in buys:
                success = self._trading_env.buy(buy["operation"])
                self.detail_tracker.track(day, success, buy)

            self.overall_tracker.track(day)

            self._trading_env.reset_day()
