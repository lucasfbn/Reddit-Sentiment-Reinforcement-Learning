from datetime import datetime
import time

import pandas as pd
import requests
from preprocess.sentiment_analysis.api.google_cloud import BigQueryDB
from psaw import PushshiftAPI

from utils import dt_to_timestamp, log, submission_schema


class MainApi:
    # Used for historic data

    def __init__(self):
        self.api = PushshiftAPI()

    def _submission_request(self, after, subreddit, end):
        if end is None:
            submissions = self.api.search_submissions(after=after, subreddit=subreddit)
        else:
            submissions = self.api.search_submissions(after=after, before=end, subreddit=subreddit)
        return submissions

    def _extract_relevant_data(self, submissions):
        new_submissions = []
        for subm in submissions:
            subm = subm.d_
            temp = {}
            for schema_key in submission_schema:
                if schema_key in subm:
                    temp[schema_key] = subm[schema_key]
                else:
                    temp[schema_key] = None
            new_submissions.append(temp)
        return new_submissions

    def _to_df(self, submissions):
        df = pd.DataFrame(submissions)
        df = df.astype(submission_schema)
        return df

    def get_submissions(self, start, subreddit, end=None):
        start, end = dt_to_timestamp(start), dt_to_timestamp(end)
        submissions = self._submission_request(start, subreddit, end=end)
        submissions = self._extract_relevant_data(submissions)
        return self._to_df(submissions)


class BetaAPI:
    # Used for current data

    def __init__(self):
        self.submissions = None

    def submission_request(self, subreddit):
        baseurl = "https://beta.pushshift.io/search/reddit/submissions"
        params = {
            "subreddit": subreddit,
            "size": 1000
        }

        response = requests.get(baseurl, params=params)

        timeout = 0

        while response.status_code != 200:
            log.warning(f"BETA API: Response status code was {response.status_code}. Will retry in {1 + timeout}.")
            response = requests.get(baseurl, params=params)
            time.sleep(1 + timeout)
            timeout += 5

        return response.json()["data"]

    def filter_early_submissions(self, start):
        filtered_submissions = []
        for submission in self.submissions:
            if submission["created_utc"] >= start:
                filtered_submissions.append(submission)

        self.submissions = filtered_submissions

    def _extract_relevant_data(self):
        new_submissions = []
        for subm in self.submissions:
            temp = {}
            for schema_key in submission_schema:
                if schema_key in subm:
                    temp[schema_key] = subm[schema_key]
                else:
                    temp[schema_key] = None
            new_submissions.append(temp)
        self.submissions = new_submissions

    def _to_df(self):
        df = pd.DataFrame(self.submissions)
        df = df.astype(submission_schema)
        return df

    def get_submissions(self, start, subreddit):
        after = dt_to_timestamp(start)
        self.submission_request(subreddit)
        self.filter_early_submissions(start)
        df = self._to_df()
        return df


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
    res = p.get_submissions(start=start, subreddit="wallstreetbets")

    # p = MainApi()
    # df = p.get_submissions(start=1614067200, end=1614088800, subreddit="stocks")
    # print()

    # download()
