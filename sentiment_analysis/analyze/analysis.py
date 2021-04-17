import pandas as pd

pd.set_option('mode.chained_assignment', None)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from utils import *


class SubmissionsHandler:

    def __init__(self, data, **kwargs):
        self.data = data
        self.kwargs = kwargs
        self.processed_data = pd.DataFrame()

    def process(self):
        len_d = len(self.data)
        for i, d in enumerate(self.data):
            log.info(f"Processing hour {i} of {len_d}")
            submission = Submissions(**self.kwargs,
                                     run_id=d["id"],
                                     df=d["df"],
                                     start=d["start"],
                                     start_timestamp=d["start_timestamp"],
                                     end=d["end"],
                                     end_timestamp=d["end_timestamp"],
                                     subreddit=d["subreddit"])
            result = submission.analyze()
            self.processed_data = self.processed_data.append(result)

        return self.processed_data


class Submissions:

    def __init__(self,
                 run_id,
                 start,
                 start_timestamp,
                 end,
                 end_timestamp,
                 subreddit,
                 search_ticker_in_body,
                 ticker_blacklist,
                 body_col,
                 cols_in_vader_merge,
                 path=None,
                 df=None):

        if path is None and df is None:
            raise ValueError("Specify either a path or a df.")

        if path is not None:
            self.df = pd.read_csv(path, sep=";")
        else:
            self.df = df

        self.run_id = run_id
        self.start, self.end = start, end
        self.start_timestamp, self.end_timestamp = start_timestamp, end_timestamp
        self.subreddit = subreddit
        self.search_ticker_in_body = search_ticker_in_body
        self.ticker_blacklist = ticker_blacklist
        self.body_col = body_col
        self.cols_in_vader_merge = cols_in_vader_merge

        self.valid_ticker = self._get_valid_ticker()
        self.submission_ticker = pd.DataFrame()
        self.ticker_aggregated = pd.DataFrame()

    def _get_valid_ticker(self):
        ticker = pd.read_csv(paths.all_ticker, sep=";")["Symbol"]
        ticker = ticker[ticker.str.len() >= 2]
        return ticker.values.tolist()

    def _extract_ticker(self, txt):
        occurred_ticker = []
        try:
            words = txt.split(" ")
        except (TypeError, AttributeError):
            return occurred_ticker

        for word in words:
            if len(word) <= 5 and word not in self.ticker_blacklist and word in self.valid_ticker:
                occurred_ticker.append(word)

        return occurred_ticker

    @drop_stats
    def _filter_no_ticker(self):
        submission_ticker_id = self.submission_ticker["id"].values.tolist()
        self.df = self.df[self.df["id"].isin(submission_ticker_id)]

    def get_submission_ticker(self):
        df_dict = self.df.to_dict("records")

        submission_ticker = []
        for i, d in enumerate(df_dict):
            log.debug(f"Processing submission {i + 1}/{len(df_dict)}")
            id = d["id"]

            occurred_ticker = self._extract_ticker(d["title"])
            if not occurred_ticker and self.search_ticker_in_body and d[self.body_col] != "nan":
                occurred_ticker = self._extract_ticker(d[self.body_col])
            occurred_ticker = list(set(occurred_ticker))

            for ot in occurred_ticker:
                submission_ticker.append({"id": id, "ticker": ot})

        self.submission_ticker = pd.DataFrame(submission_ticker)

    def _map_polarity_to_ticker(self):
        vader = self.df[self.cols_in_vader_merge]
        merged = self.submission_ticker.merge(vader, on="id")
        return merged

    def apply_vader(self):
        analyzer = SentimentIntensityAnalyzer()

        def wrapper(row):
            scores = analyzer.polarity_scores(row["title"])
            return [scores["pos"], scores["neu"], scores["neg"], scores["compound"]]

        self.df[["pos", "neu", "neg", "compound"]] = self.df.apply(wrapper, axis=1, result_type="expand")
        self.submission_ticker = self._map_polarity_to_ticker()

    def _get_n_posts(self):
        grps_count = self.submission_ticker[["id", "ticker"]].groupby("ticker").count().reset_index()
        return grps_count.rename(columns={"id": "n_posts"})

    def add_metadata(self):
        self.ticker_aggregated["run_id"] = self.run_id
        self.ticker_aggregated["start"] = self.start
        self.ticker_aggregated["start_timestamp"] = self.start_timestamp

        self.ticker_aggregated["end"] = self.end
        self.ticker_aggregated["end_timestamp"] = self.end_timestamp

        self.ticker_aggregated["subreddit"] = self.subreddit

    def aggregate(self):
        n_posts = self._get_n_posts()
        grps = self.submission_ticker.groupby("ticker").agg("sum").reset_index()
        grps = grps.merge(n_posts, on="ticker")
        self.ticker_aggregated = grps

    def analyze(self):
        self.get_submission_ticker()

        if self.submission_ticker.empty:
            return pd.DataFrame()

        self._filter_no_ticker()
        self.apply_vader()
        self.aggregate()
        self.add_metadata()
        return self.ticker_aggregated


if __name__ == "__main__":
    with open("raw.pkl", "rb") as f:
        data = pkl.load(f)

    sh = SubmissionsHandler(data)
    p_data = sh.process()
    p_data.to_csv("report.csv", sep=";", index=False)
    print()
