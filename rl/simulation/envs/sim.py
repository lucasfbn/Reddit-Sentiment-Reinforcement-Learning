import itertools

import pandas as pd
import wandb
from tqdm import tqdm

from rl.simulation.envs.sub_envs.trading import TradingSimulator
from rl.simulation.envs.tracker.track import Tracker, EnvStateTracker
from utils.wandb_utils import log_to_summary


class Simulation:

    def __init__(self, dataset):
        self.dataset = dataset

        self._daywise_sequences = None
        self._trading_env = TradingSimulator()

        self.detail_tracker = Tracker(self._trading_env)
        self.overall_tracker = EnvStateTracker(self._trading_env)

    def prepare_data(self):
        sequences = [list(t.sequences) for t in self.dataset]
        sequences = list(itertools.chain(*sequences))

        sequences = [
            seq
            for seq in sequences
            if seq.evl.action != 1 or seq.portfolio.execute
        ]

        for seq in sequences:
            seq.metadata.date = pd.Period(seq.metadata.date)

        sequences = sorted(sequences, key=lambda seq: seq.metadata.date)

        daywise = {
            key: list(group)
            for key, group in itertools.groupby(
                sequences, lambda seq: seq.metadata.date
            )
        }

        self._daywise_sequences = daywise
        return daywise

    def per_action_callback(self, day, success, sequence):
        pass

    def end_of_day_callback(self, day):
        pass

    def end_of_eval_callback(self):
        pass

    def eval_loop(self):

        for day, sequences in tqdm(self._daywise_sequences.items(), desc="Processing day"):

            actions = {"0": [], "1": [], "2": []}

            # Required for groupby
            sequences = sorted(sequences, key=lambda seq: seq.evl.action)

            for key, group in itertools.groupby(sequences, lambda seq: seq.evl.action):
                actions[str(key)] = list(group)

            assert list(actions.keys()) == ["0", "1", "2"]

            holds, buys, sells = actions["0"], actions["1"], actions["2"]

            # Execute sells first
            for sell in sells:
                success = self._trading_env.sell(sell)
                self.per_action_callback(day, success, sell)

            for hold in holds:
                success = self._trading_env.hold(hold)
                self.per_action_callback(day, success, hold)

            for buy in buys:
                success = self._trading_env.buy(buy)
                self.per_action_callback(day, success, buy)

            self.end_of_day_callback(day)
            self._trading_env.reset_day()

        self.end_of_eval_callback()


class SimulationWandb(Simulation):

    def end_of_day_callback(self, day):
        wandb.log(dict(day=str(day),
                       inventory_len=len(self._trading_env.inventory),
                       balance=self._trading_env.balance))

    def end_of_eval_callback(self):
        log_to_summary(wandb.run, dict(final_balance=self._trading_env.balance,
                                       profit=self._trading_env.balance / self._trading_env.START_BALANCE,
                                       inventory_len=len(self._trading_env.inventory)))
