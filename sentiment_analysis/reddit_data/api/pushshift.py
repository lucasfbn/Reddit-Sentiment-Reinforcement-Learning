import time
from datetime import datetime

import requests
from psaw import PushshiftAPI

from sentiment_analysis.reddit_data.api.api import API
from sentiment_analysis.reddit_data.api.google_cloud import BigQueryDB
from utils.util_funcs import dt_to_timestamp, log


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

    def extract_relevant_data(self, submission):
        temp = {}
        submission_dict = submission._asdict()
        for schema_key in self.submission_schema:
            if schema_key in submission_dict:
                if schema_key == "author":
                    try:
                        author_name = submission.author
                    except:
                        author_name = None
                    temp[schema_key] = author_name
                elif schema_key == "subreddit":
                    temp[schema_key] = submission.subreddit
                else:
                    temp[schema_key] = submission_dict[schema_key]
            else:
                temp[schema_key] = None
        return temp

    def get_submissions(self, start, subreddit, filter_removed, end=None):
        start, end = dt_to_timestamp(start), dt_to_timestamp(end)
        self.submissions = self._submission_request(start, subreddit, end=end)

        new_submissions = []
        for subm in self.submissions:
            new_submissions.append(self.extract_relevant_data(subm))

        self.submissions = new_submissions
        df = self.to_df()
        if filter_removed:
            df = self._filter_removed(df)
        return df

    def get_submission_ids(self, start, end, subreddit, filter_removed):
        df = self.get_submissions(start=start, end=end, subreddit=subreddit, filter_removed=filter_removed)
        return df["id"].values.tolist()


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

    def extract_relevant_data(self):
        new_submissions = []
        for subm in self.submissions:
            temp = {}
            for schema_key in self.submission_schema:
                if schema_key in subm:
                    temp[schema_key] = subm[schema_key]
                else:
                    temp[schema_key] = None
            new_submissions.append(temp)
        self.submissions = new_submissions

    def get_submissions(self, start, subreddit, filter_removed):
        start = dt_to_timestamp(start)
        self.submissions, _ = self._submission_request(subreddit)
        self._filter_early_submissions(start)
        self.extract_relevant_data()
        df = self.to_df()

        if filter_removed:
            df = self._filter_removed(df)

        return df

    def get_submission_ids(self, start, subreddit, filter_removed):
        df = self.get_submissions(start, subreddit, filter_removed)
        return df["id"].values.tolist()


class ManualDownload(BetaAPI):

    def __init__(self, data):
        self.submissions = data

    def get_submissions(self):
        self.extract_relevant_data()
        return self.to_df()


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
    import json

    from utils import paths

    start = datetime(year=2021, month=2, day=23, hour=21)

    with open(paths.sentiment_data_path / "01-11-21 - 30-11-21_0_MANUAL" / "data.json") as f:
        data = json.load(f)
    md = ManualDownload(data)
    df = md.get_submissions()
    df.to_csv(paths.sentiment_data_path / "01-11-21 - 30-11-21_0_MANUAL" / "gc_dump.csv", sep=";")
