import os
import pickle as pkl

import pandas as pd

from preprocess.sentiment_analysis.analyze.analysis import SubmissionsHandler
from preprocess.sentiment_analysis.api.download import download
from preprocess.sentiment_analysis.preprocess.preprocess import Preprocessor
import paths


class Dataset:
    gc_dump_fn = "gc_dump.csv"
    grps_fn = "grps.pkl"
    report_fn = "report.csv"

    def __init__(self, start, end,
                 fields=["author", "created_utc", "id", "num_comments", "title", "selftext", "subreddit"],
                 path_suffix="",
                 upload_report=False):
        self.start = start
        self.end = end
        self.fields = fields
        self.upload_report = upload_report
        self.path = paths.sentiment_data_path / path_suffix

        self._prepare_dir()

        self.df = None
        self.grps = None
        self.report = None

    def _prepare_dir(self):
        n_folder = len([_ for _ in os.listdir(self.path) if os.path.isdir(self.path / _)])
        fn = f"{self.start.strftime('%d-%m-%y')} - {self.end.strftime('%d-%m-%y')}"
        self.path = paths.create_dir(self.path, fn, 0)

    def get_from_gc(self):
        self.df = download(self.start, self.end, self.fields)
        self.df.to_csv(self.path / self.gc_dump_fn, sep=";", index=False)

    def preprocess(self):
        if self.df is None:
            self.df = pd.read_csv(self.path / self.gc_dump_fn, sep=";")

        prep = Preprocessor(df=self.df)
        self.grps = prep.run()

        with open(self.path / self.grps_fn, "wb") as f:
            pkl.dump(self.grps, f)

    def analyze(self):
        if self.grps is None:
            with open(self.path / self.grps_fn, "rb") as f:
                self.grps = pkl.load(f)

        sh = SubmissionsHandler(data=self.grps,
                                upload=self.upload_report,
                                upload_all_at_once=False,
                                search_ticker_in_body=True,
                                max_subm_p_author_p_day=1)
        p_data = sh.process()
        p_data.to_csv(self.path / self.report_fn, sep=";", index=False)

    def create(self):
        self.get_from_gc()
        self.preprocess()
        self.analyze()


if __name__ == "__main__":
    from datetime import datetime

    start = datetime(year=2021, month=2, day=14)
    end = datetime(year=2021, month=2, day=24)
    ds = Dataset(start, end, path_suffix="live")
    ds.create()
