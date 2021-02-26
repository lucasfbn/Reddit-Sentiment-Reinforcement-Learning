from datetime import datetime
import time

import pandas as pd
import requests
from preprocess.sentiment_analysis.api.google_cloud import BigQueryDB
from preprocess.sentiment_analysis.api.api import API
from psaw import PushshiftAPI

from utils import dt_to_timestamp, log


class MainApi(API):
    # Used for historic data

    def __init__(self):
        self.api = PushshiftAPI()
        self.submissions = None

    def _submission_request(self, after, subreddit, end):
        if end is None:
            submissions = self.api.search_submissions(after=after, subreddit=subreddit)
        else:
            submissions = self.api.search_submissions(after=after, before=end, subreddit=subreddit)
        return submissions

    def get_submissions(self, start, subreddit, end=None):
        start, end = dt_to_timestamp(start), dt_to_timestamp(end)
        self.submissions = self._submission_request(start, subreddit, end=end)
        self.extract_relevant_data()
        return self.to_df()


class BetaAPI(API):
    # Used for current data

    def __init__(self, max_retries=5):
        self.max_retries = max_retries

        self.submissions = None

    def available(self):
        _, status_code = self._submission_request(subreddit="pushshift", size=10, max_retries=1)
        if status_code == 200:
            return True
        return False

    def _submission_request(self, subreddit, size=1000, max_retries=None):
        baseurl = "https://beta.pushshift.io/search/reddit/submissions"
        params = {
            "subreddit": subreddit,
            "size": size
        }

        response = requests.get(baseurl, params=params)

        retries = 0
        if max_retries is None:
            max_retries = self.max_retries

        while response.status_code != 200 and retries < max_retries:
            log.warning(f"BETA API: Response status code was {response.status_code}. Will retry in {1}.")
            response = requests.get(baseurl, params=params)
            time.sleep(1)
            retries += 1

        if response.status_code == 200:
            return response.json()["data"], response.status_code
        else:
            return [], response.status_code

    def _filter_early_submissions(self, start):
        filtered_submissions = []
        for submission in self.submissions:
            if submission["created_utc"] >= start:
                filtered_submissions.append(submission)

        self.submissions = filtered_submissions

    def _filter_removed(self, df):
        cols_to_check_if_removed = ["author", "selftext", "title"]
        for col in cols_to_check_if_removed:
            df = df[~df[col].isin(["[removed]", "[deleted]"])]
        return df

    def get_submissions(self, start, subreddit):
        start = dt_to_timestamp(start)
        self.submissions, _ = self._submission_request(subreddit)
        self._filter_early_submissions(start)
        self.extract_relevant_data()
        df = self.to_df()
        return df

    def get_submission_ids(self, start, subreddit, filter_removed):
        df = self.get_submissions(start, subreddit)
        if filter_removed:
            df = self._filter_removed(df)

        return df["id"].values.tolist()


def download():
    api = MainApi()
    db = BigQueryDB()

    start = datetime(year=2021, month=2, day=23, hour=9)
    end = datetime(year=2021, month=2, day=25)

    subreddits = [
        "pennystocks",
        "RobinHoodPennyStocks",
        "Daytrading",
        "StockMarket",
        "stocks",
        "trakstocks",
        "SPACs",
        "wallstreetbets",
    ]

    for i, subreddit in enumerate(subreddits):
        # log.info(f"Processing {subreddit}. {i + 1}/{len(subreddits)}")
        submissions = api.get_submissions(start=start, end=end, subreddit=subreddit)
        db.upload(submissions, dataset="data", table="submissions")


if __name__ == "__main__":
    start = datetime(year=2021, month=2, day=23, hour=21)

    p = BetaAPI()
    print(p.available())
    res = p.get_submissions(start=start, subreddit="wallstreetbets")

    # p = MainApi()
    # df = p.get_submissions(start=1614067200, end=1614088800, subreddit="stocks")
    # print()

    # download()
    print()
