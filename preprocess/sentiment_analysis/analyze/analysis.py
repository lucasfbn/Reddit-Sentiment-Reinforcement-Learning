import pickle as pkl

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import paths
from preprocess.sentiment_analysis.db.db_handler import DB
from preprocess.sentiment_analysis.utils.utils import *


class SubmissionsHandler:

    def __init__(self, data, upload=False, upload_all_at_once=True, **kwargs):
        self.data = data
        self.upload = upload
        self.upload_all_at_once = upload_all_at_once
        self.kwargs = kwargs
        self.processed_data = pd.DataFrame()

    def process(self):
        len_d = len(self.data)
        for i, d in enumerate(self.data):
            log.info(f"Processing {i}/{len_d}")
            submission = Submissions(**self.kwargs,
                                     run_id=d["id"],
                                     df=d["df"],
                                     start=d["start"],
                                     start_timestamp=d["start_timestamp"],
                                     end=d["end"],
                                     end_timestamp=d["end_timestamp"],
                                     subreddit=d["subreddit"])
            result = submission.process()
            if not self.upload_all_at_once:
                self._upload(result)
            self.processed_data = self.processed_data.append(result)

        if self.upload_all_at_once:
            self._upload()

        return self.processed_data

    def _upload(self, data=None):

        if not self.upload:
            return

        if data is None:
            data = self.processed_data

        log.info(f"Uploading... Rows: {len(data)}")
        database = DB()
        database.up(data, "processed_data", "ticker")


class Submissions:
    ticker_blacklist = ["DD"]
    author_blacklist = []

    def __init__(self,
                 run_id,
                 start,
                 start_timestamp,
                 end,
                 end_timestamp,
                 subreddit,
                 path=None,
                 df=None,
                 max_subm_p_author_p_day=1,
                 search_ticker_in_body=True):

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
        self.max_subm_p_author_p_day = max_subm_p_author_p_day
        self.search_ticker_in_body = search_ticker_in_body

        self.cols_to_be_cleaned = ["title"]
        self.body_col = "selftext"
        self.cols_in_vader_merge = ["id", "num_comments", "date", "pos", "compound", "neu", "neg", "date_mesz"]

        self.valid_ticker = self._get_valid_ticker()
        self.submission_ticker = pd.DataFrame()
        self.ticker_aggregated = pd.DataFrame()

    def _get_valid_ticker(self):
        ticker = pd.read_csv(paths.all_ticker, sep=";")["Symbol"]
        ticker = ticker[ticker.str.len() >= 2]
        return ticker.values.tolist()

    @drop_stats
    def _filter_authors(self):
        filtered_df = pd.DataFrame()

        grps = self.df.groupby(["date"])

        for i, (name, grp) in enumerate(grps):
            grp = pd.DataFrame(grp)
            authors = grp.groupby(["author"])

            for j, (author_id, author_grp) in enumerate(authors):
                log.debug(f"Processing grp: {i}/{len(grps)}. Author: {j}/{len(authors)}")

                if author_id in self.author_blacklist:
                    continue

                if len(author_grp) == self.max_subm_p_author_p_day:
                    filtered_df = filtered_df.append(author_grp)
                else:
                    author_grp = author_grp.sort_values(by=["num_comments"], ascending=False)
                    filtered_df = filtered_df.append(author_grp.head(self.max_subm_p_author_p_day))

        self.df = filtered_df

    def _delete_non_alphanumeric(self):
        for col in self.cols_to_be_cleaned:
            self.df[col] = self.df[col].str.replace('[^\w\s,.?!()-+:"]', '', regex=True)

    def preprocess(self):
        self._filter_authors()
        self._delete_non_alphanumeric()

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
        self._filter_no_ticker()

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

    def process(self):
        self.preprocess()
        self.get_submission_ticker()
        self.apply_vader()
        self.aggregate()
        self.add_metadata()
        return self.ticker_aggregated


if __name__ == "__main__":
    with open("raw.pkl", "rb") as f:
        data = pkl.load(f)

    sh = SubmissionsHandler(data,
                            upload=False,
                            upload_all_at_once=False)
    p_data = sh.process()
    p_data.to_csv("report.csv", sep=";", index=False)
    print()
