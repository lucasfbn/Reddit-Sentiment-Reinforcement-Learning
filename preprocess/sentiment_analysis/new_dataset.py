import os
import pickle as pkl

import pandas as pd

from analyze.analysis import SubmissionsHandler
from download.from_gc.download import download
from preprocess.preprocess import Preprocessor


class Dataset:
    gc_dump_fn = "gc_dump.csv"
    grps_fn = "grps.pkl"
    report_fn = "report.csv"

    def __init__(self, start, end,
                 fields=["author", "created_utc", "id", "num_comments", "title", "selftext", "subreddit"],
                 upload_report=False):
        self.start = start
        self.end = end
        self.fields = fields
        self.upload_report = upload_report

        self._prepare_dir()

        self.path = None

        self.df = None
        self.grps = None
        self.report = None

    def _prepare_dir(self):
        n_folder = len([_ for _ in os.listdir() if os.path.isdir(_)])
        new_dir = f"{n_folder + 1}"
        os.mkdir(new_dir)
        self.path = new_dir

    def get_from_gc(self):
        self.df = download(self.start, self.end, self.fields)
        self.df.to_csv(self.path / self.gc_dump_fn, sep=";", index=False)

    def preprocess(self):
        if self.df is None:
            self.df = pd.read_csv(self.path / self.gc_dump_fn, sep=";")

        prep = Preprocessor(self.df)
        self.grps = prep.run()

        with open(self.path / self.grps_fn, "wb") as f:
            pkl.dump(self.grps, f)

    def analyze(self):
        if self.grps is None:
            with open(self.path / self.grps_fn, "rb") as f:
                self.grps = pkl.load(f)

        sh = SubmissionsHandler(self.grps,
                                upload=self.upload_report,
                                upload_all_at_once=False)
        p_data = sh.process()
        p_data.to_csv(self.path / self.report_fn, sep=";", index=False)

    def create(self):
        self.get_from_gc()
        self.preprocess()
        self.analyze()


if __name__ == "__main__":
    from datetime import datetime

    start = datetime(year=2021, month=1, day=13)
    end = datetime(year=2021, month=1, day=13, hour=1)
    ds = Dataset(start, end)
    ds.create()
