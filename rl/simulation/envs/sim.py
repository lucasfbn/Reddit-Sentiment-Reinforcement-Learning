import itertools

import pandas as pd
import wandb
from tqdm import tqdm

from rl.simulation.envs.sub_envs.trading import TradingSimulator
from utils.wandb_utils import log_to_summary, log_file


class Simulation:

    def __init__(self, dataset):
        self.dataset = dataset

        self._daywise_sequences = None
        self._trading_env = TradingSimulator()

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

    def start_of_day_callback(self, day):
        pass

    def end_of_day_callback(self, day_index, day, actions):
        pass

    def end_of_eval_callback(self):
        pass

    def eval_loop(self):

        for i, (day, sequences) in tqdm(enumerate(self._daywise_sequences.items()), desc="Processing day"):

            self.start_of_day_callback(i)

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

            self.end_of_day_callback(i, day, actions)
            self._trading_env.reset_day()

        self.end_of_eval_callback()


class SimulationWandb(Simulation):

    def __init__(self, dataset):
        super().__init__(dataset)

        self.tracked = []

    def per_action_callback(self, day, success, sequence):
        self.tracked.append(dict(
            ticker=sequence.metadata.ticker_name,
            date=str(day),
            success=success,
            action=sequence.evl.action,
            exec=bool(sequence.portfolio.execute),
            tradeable=sequence.metadata.tradeable,
            price=sequence.metadata.price_raw
        ))

    def start_of_day_callback(self, day_index):
        wandb.log(dict(day=day_index,
                       inventory_len=len(self._trading_env.inventory),
                       balance=self._trading_env.balance,
                       balance_rel=self._trading_env.balance / self._trading_env.START_BALANCE))

    def end_of_day_callback(self, day_index, day, actions):
        holds, buys, sells = actions["0"], actions["1"], actions["2"]
        total = len(holds) + len(buys) + len(sells)

        wandb.log(dict(day=day_index,
                       buy_ratio=len(buys) / total,
                       hold_ratio=len(holds) / total,
                       sell_ratio=len(sells) / total))

    def end_of_eval_callback(self):
        log_to_summary(wandb.run, dict(final_balance=self._trading_env.balance,
                                       profit=self._trading_env.balance / self._trading_env.START_BALANCE,
                                       inventory_len=len(self._trading_env.inventory)))

        df = pd.DataFrame(self.tracked)
        df = df[df["action"] != 0]
        log_file(df, "tracked.csv", wandb.run)
