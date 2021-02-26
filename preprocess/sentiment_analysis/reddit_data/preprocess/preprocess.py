import pickle as pkl

import pandas as pd

from utils import *


class Preprocessor:

    def __init__(self,
                 path=None,
                 df=None):
        if path is None and df is None:
            raise ValueError("Specify either a path or a df.")

        if path is not None:
            self.df = pd.read_csv(path, sep=";")
        else:
            self.df = df

        self.cols_to_check_if_removed = ["author", "selftext", "title"]
        report.add({"cols_to_check_if_removed": self.cols_to_check_if_removed}, "Preprocessor")

    def _add_date_col(self):
        self.df["date_full"] = pd.to_datetime(self.df["created_utc"], unit="s")

        self.df["date_full"] = self.df["date_full"].dt.tz_localize("UTC")
        self.df["date_mesz"] = self.df["date_full"].dt.tz_convert("Europe/Berlin")
        self.df["start"] = self.df["date_mesz"].dt.to_period('H')
        self.df["start_timestamp"] = self.df["date_mesz"].astype(int) / 10 ** 9

        end = self.df["date_mesz"] + pd.Timedelta(seconds=3600)
        self.df["end"] = end.dt.to_period('H')
        self.df["end_timestamp"] = end.astype(int) / 10 ** 9

        self.df["date"] = self.df["date_mesz"].dt.to_period('D')

    @drop_stats
    def _filter_removed(self):
        for col in self.cols_to_check_if_removed:
            self.df = self.df[~self.df[col].isin(["[removed]", "[deleted]"])]
        report.add({"filter_removed": True}, "Preprocessor")

    def _split_by_date(self):
        grps = self.df.groupby("start")

        new_grps = []
        for name, grp in grps:
            # # TODO Add subreddit groupby and respect in analysis?
            # subreddit_grp = grp.groupby("subreddit")

            # for subr, subr_grp in subreddit_grp:
            new_grps.append({"id": str(name), "df": grp,
                             "start": grp["start"].iloc[0], "end": grp["end"].iloc[0],
                             "start_timestamp": grp["start_timestamp"].iloc[0],
                             "end_timestamp": grp["end_timestamp"].iloc[0],
                             "subreddit": "all"})

        return new_grps

    def run(self):
        self._filter_removed()
        self._add_date_col()
        grps = self._split_by_date()
        return grps


if __name__ == '__main__':
    prep = Preprocessor("../analyze/raw.csv")
    grps = prep.run()
    with open("../analyze/raw.pkl", "wb") as f:
        pkl.dump(grps, f)
