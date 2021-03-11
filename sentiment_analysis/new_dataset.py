import os
import pickle as pkl

import pandas as pd

import paths
import sentiment_analysis.config as config
from sentiment_analysis.analyze.analysis import SubmissionsHandler
from sentiment_analysis.reddit_data.api.google_cloud import BigQueryDB
from sentiment_analysis.reddit_data.preprocess.preprocess import Preprocessor
from utils import save_config


class Dataset:
    gc_dump_fn = "gc_dump.csv"
    grps_fn = "grps.pkl"
    report_fn = "report.csv"

    def __init__(self):
        self.start = config.general.start
        self.end = config.general.end
        self.check_integrity = config.check.integrity

        if config.general.path is None:
            self.path = paths.sentiment_data_path
            self._prepare_dir()
        else:
            self.path = config.general.path

        self.df = None
        self.grps = None
        self.report = None

    def _prepare_dir(self):
        n_folder = len([_ for _ in os.listdir(self.path) if os.path.isdir(self.path / _)])
        fn = f"{self.start.strftime('%d-%m-%y')} - {self.end.strftime('%d-%m-%y')}"
        self.path = paths.create_dir(self.path, fn, 0)

    def get_from_gc(self):
        if self.df is None:
            try:
                self.df = pd.read_csv(self.path / self.gc_dump_fn, sep=";")
            except FileNotFoundError:
                db = BigQueryDB()
                self.df = db.download(self.start, self.end, config.gc.fields)
                self.df.to_csv(self.path / self.gc_dump_fn, sep=";", index=False)

    def preprocess(self):
        if self.df is None:
            self.df = pd.read_csv(self.path / self.gc_dump_fn, sep=";")

        prep = Preprocessor(df=self.df,
                            author_blacklist=config.preprocess.author_blacklist,
                            cols_to_check_if_removed=config.preprocess.cols_to_check_if_removed,
                            cols_to_be_cleaned=config.preprocess.cols_to_be_cleaned,
                            max_subm_p_author_p_day=config.preprocess.max_subm_p_author_p_day,
                            filter_authors=config.preprocess.filter_authors)
        self.grps = prep.exec()

        with open(self.path / self.grps_fn, "wb") as f:
            pkl.dump(self.grps, f)

    def _check_integrity(self):

        if not self.check_integrity:
            return

        if self.grps is None:
            with open(self.path / self.grps_fn, "rb") as f:
                self.grps = pkl.load(f)
        daterange = pd.date_range(start=self.start, end=self.end - pd.Timedelta(hours=1), freq="H")

        if len(daterange) != len(self.grps):
            raise ValueError(
                f"The dataset is missing entries. len daterange: {len(daterange)}, len grps: {len(self.grps)}.")
        for grp in self.grps:
            if len(grp["df"]) == 0:
                raise ValueError(f"Df of grp {grp['id']} is 0.")

    def analyze(self):
        if self.grps is None:
            with open(self.path / self.grps_fn, "rb") as f:
                self.grps = pkl.load(f)

        sh = SubmissionsHandler(data=self.grps,
                                search_ticker_in_body=config.submissions.search_ticker_in_body,
                                ticker_blacklist=config.submissions.ticker_blacklist,
                                body_col=config.submissions.body_col,
                                cols_in_vader_merge=config.submissions.cols_in_vader_merge)
        p_data = sh.process()
        p_data.to_csv(self.path / self.report_fn, sep=";", index=False)

    def create(self):
        self.get_from_gc()
        self.preprocess()
        self._check_integrity()
        self.analyze()
        
        config.general.path = self.path
        save_config([config.general, config.gc, config.preprocess, config.check, config.submissions], kind="sentiments")


if __name__ == "__main__":
    ds = Dataset()
    # ds.df = pd.read_csv(path / "gc_dump.csv", sep=";")
    ds.create()
