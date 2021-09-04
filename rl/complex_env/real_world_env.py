import mlflow
import pandas as pd

import paths
from utils.mlflow_api import load_file
from rl.complex_env.sim import Sim


class RealWorldEnv:

    def __init__(self, ticker):
        self.ticker = ticker
        self.ticker_df = None
        self.sequences_daywise = None

        self.curr_date_idx = 0

        self.episode_end = False
        self.episode_count = 0

        self.sim_env = Sim()

    def create_ticker_df(self):
        dicts = []

        for i, ticker in enumerate(self.ticker):
            for j, seq in enumerate(ticker.sequences):
                seq_dict = dict(
                    ticker_id=i,
                    sequence_id=j,
                    ticker=ticker.name,
                    price=seq.price_raw,
                    date=seq.date,
                    tradeable=seq.tradeable,
                )
                dicts.append(seq_dict)

        self.ticker_df = pd.DataFrame(dicts)

    def daywise(self):
        dates = []
        # Group by date, convert grp rows to Operations and add them to the dates dict
        self.ticker_df = self.ticker_df.sort_values(by=["date"])
        grps = self.ticker_df.groupby(["date"])

        for name, grp in grps:
            lst = []

            def add_ticker_seq(row):
                tck = self.ticker[row["ticker_id"]]
                seq = tck.sequences[row["sequence_id"]]
                seq.ticker_name = tck.name
                lst.append(seq)

            pd.DataFrame(grp).apply(add_ticker_seq, axis="columns")

            dates.append(lst)

        self.sequences_daywise = dates

    def initialize(self):
        self.create_ticker_df()
        self.daywise()

    def next_date(self):
        if self.curr_date_idx == len(self.sequences_daywise):
            self.curr_date_idx = 0
            self.episode_end = True
            return None
        else:
            r = self.sequences_daywise[self.curr_date_idx]
            self.curr_date_idx += 1
        return r

    def reset(self):
        self.episode_end = False
        self.episode_count += 1

        self.sim_env = Sim()

        return self.next_date()

    def execute(self, states):
        # reward = self.sim_env.process_sequences(states)
        next_states = self.next_date()
        reward = 0

        return next_states, self.episode_end, reward

    def max_episode_timesteps(self):
        return max(len(day) for day in self.sequences_daywise)


class MockObj(object):
    def __init__(self, **kwargs):  # constructor turns keyword args into attributes
        self.__dict__.update(kwargs)


if __name__ == '__main__':
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Tests")
    with mlflow.start_run():
        data = load_file(run_id="f4bdae299f694599ba91c7dd1f77c9b5", fn="ticker.pkl", experiment="Datasets")

        ce = RealWorldEnv(ticker=data)
        ce.initialize()

        mock = MockObj(action=0)

        states = ce.reset()
        terminal = False
        for _ in range(1000):

            while not terminal:
                new_states, terminal, reward = ce.execute(mock)

                if terminal == False:
                    print("Hallo")
