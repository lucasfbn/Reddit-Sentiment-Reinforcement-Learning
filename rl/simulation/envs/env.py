from tqdm import tqdm

from rl.simulation.envs.sub_envs.trading import TradingSimulator
from rl.simulation.envs.tracker.track import Tracker, EnvStateTracker
from rl.simulation.envs.utils.utils import *


class EvalEnv:

    def __init__(self, ticker, pre_processor):
        self.ticker = ticker
        self.pre_processor = pre_processor

        self._trading_env = TradingSimulator()

        self.detail_tracker = Tracker(self._trading_env)
        self.overall_tracker = EnvStateTracker(self._trading_env)

    def run_pre_processor(self):
        self.ticker = self.pre_processor.run(self.ticker)

    def eval_loop(self):
        ticker_df = ticker_list_to_df(self.ticker)
        ordered_day_wise = order_day_wise(self.ticker, ticker_df)

        for day, operations in tqdm(ordered_day_wise.items(), desc="Processing day"):

            action_pairs = [dict(action=operation.sequence.evl.action,
                                 proba=operation.sequence.evl.probas,
                                 operation=operation) for operation in operations]

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
