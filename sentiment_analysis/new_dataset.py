import warnings

import mlflow

import paths
import sentiment_analysis.config as sentiment_analysis_config
from sentiment_analysis.analyze.analysis import SubmissionsHandler
from sentiment_analysis.reddit_data.api.google_cloud import BigQueryDB
from sentiment_analysis.reddit_data.preprocess.preprocess import Preprocessor
from utils import save_config
from mlflow_api import log_file


class Dataset:
    gc_dump_fn = "gc_dump.csv"
    grps_fn = "grps.pkl"
    report_fn = "report.csv"

    def __init__(self, config):
        self.config = config
        self.start = config.general.start
        self.end = config.general.end
        self.check_integrity = config.check.integrity

        self.df = None
        self.grps = None
        self.report = None

    def get_from_gc(self):
        db = BigQueryDB()
        self.df = db.download(self.start, self.end, self.config.gc.fields,
                              check_duplicates=self.config.gc.check_duplicates)
        log_file(self.df, self.gc_dump_fn)

    def preprocess(self):
        prep = Preprocessor(df=self.df,
                            author_blacklist=self.config.preprocess.author_blacklist,
                            cols_to_check_if_removed=self.config.preprocess.cols_to_check_if_removed,
                            cols_to_be_cleaned=self.config.preprocess.cols_to_be_cleaned,
                            max_subm_p_author_p_day=self.config.preprocess.max_subm_p_author_p_day,
                            filter_authors=self.config.preprocess.filter_authors)
        self.grps = prep.exec()

        log_file(self.grps, self.grps_fn)

    def _check_integrity(self):

        if not self.check_integrity:
            warnings.warn("Check integrity if off.")
            return

        db = BigQueryDB()
        gaps = db.detect_gaps(self.start, self.end, save_json=False)

        if len(gaps) > 1:  # One entry is always meta data
            log_file(gaps, "gaps.json")
            raise ValueError(f"The dataset is missing entries. Logged data to mlflow run.")

        for grp in self.grps:
            if len(grp["df"]) == 0:
                raise ValueError(f"Df of grp {grp['id']} is 0.")

    def analyze(self):
        sh = SubmissionsHandler(data=self.grps,
                                search_ticker_in_body=self.config.submissions.search_ticker_in_body,
                                ticker_blacklist=self.config.submissions.ticker_blacklist,
                                body_col=self.config.submissions.body_col,
                                cols_in_vader_merge=self.config.submissions.cols_in_vader_merge)
        p_data = sh.process()
        log_file(p_data, self.report_fn)

    def create(self):
        self.get_from_gc()
        self.preprocess()
        self._check_integrity()
        self.analyze()

        save_config([self.config.general, self.config.gc,
                     self.config.preprocess, self.config.check, self.config.submissions])


if __name__ == "__main__":
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Sentiment_Datasets")  #
    mlflow.start_run()

    ds = Dataset(sentiment_analysis_config)
    ds.create()

    mlflow.end_run()
