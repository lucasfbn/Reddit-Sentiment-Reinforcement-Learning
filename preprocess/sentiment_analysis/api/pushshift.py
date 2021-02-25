from datetime import datetime

import pandas as pd
import requests
from preprocess.sentiment_analysis.api.google_cloud import BigQueryDB
from psaw import PushshiftAPI

from utils import dt_to_timestamp

submission_schema = {"created_utc": "int64", "author": "object", "id": "object", "title": "object",
                     "subreddit": "object", "num_comments": "int64", "selftext": "object",
                     "score": "int64", "upvote_ratio": "float64", "full_link": "object", "author_fullname": "object",
                     "allow_live_comments": "bool", "author_flair_css_class": "object",
                     "author_flair_richtext": "object", "author_flair_text": "object", "author_flair_type": "object",
                     "author_patreon_flair": "object", "author_premium": "object", "awarders": "object",
                     "can_mod_post": "bool", "contest_mode": "bool", "domain": "object",
                     "gildings": "object", "is_crosspostable": "bool", "is_meta": "bool",
                     "is_original_content": "bool", "is_reddit_media_domain": "bool", "is_robot_indexable": "bool",
                     "is_self": "bool", "is_video": "bool", "link_flair_background_color": "object",
                     "link_flair_css_class": "object", "link_flair_richtext": "object",
                     "link_flair_template_id": "object", "link_flair_text": "object",
                     "link_flair_text_color": "object", "link_flair_type": "object", "locked": "bool",
                     "media_only": "bool", "no_follow": "bool", "num_crossposts": "int64", "over_18": "bool",
                     "parent_whitelist_status": "object", "permalink": "object", "pinned": "bool",
                     "post_hint": "object", "removed_by_category": "object", "retrieved_on": "int64",
                     "send_replies": "bool", "spoiler": "bool", "stickied": "bool",
                     "subreddit_id": "object", "subreddit_subscribers": "int64", "subreddit_type": "object",
                     "suggested_sort": "object", "thumbnail": "object", "thumbnail_height": "float64",
                     "thumbnail_width": "float64", "total_awards_received": "int64", "treatment_tags": "object",
                     "url": "object", "url_overridden_by_dest": "object",
                     "whitelist_status": "object", "created": "float64",
                     "author_flair_background_color": "object", "author_flair_text_color": "object", "media": "object",
                     "media_embed": "object", "secure_media": "object", "secure_media_embed": "object",
                     "is_gallery": "object", "gallery_data": "object", "author_flair_template_id": "object",
                     "author_cakeday": "object", "banned_by": "object", "distinguished": "object", "edited": "float64",
                     "gilded": "object", "collections": "object"}


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

    def submission_request(self, subreddit):
        baseurl = "https://beta.pushshift.io/search/reddit/submissions"
        params = {
            "subreddit": subreddit,
            "size": 1000
        }

        response = requests.get(baseurl, params=params)
        assert response.status_code == 200
        return response.json()["data"]

    def filter_early_submissions(self, submissions, start):
        filtered_submissions = []
        for submission in submissions:
            if submission["created_utc"] >= start:
                filtered_submissions.append(submission)

        return filtered_submissions

    def _extract_ids(self, submissions):
        ids = []
        for subm in submissions:
            ids.append(subm["id"])
        return ids

    def get_submissions_ids(self, start, subreddit):
        after = dt_to_timestamp(start)
        submissions = self.submission_request(subreddit)
        submissions = self.filter_early_submissions(submissions, start)
        submission_ids = self._extract_ids(submissions)
        return submission_ids


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
    # p = BetaAPI()
    # res = p.get_submissions_ids(start=1614272760, subreddit="wallstreetbets")

    # p = MainApi()
    # df = p.get_submissions(start=1614067200, end=1614088800, subreddit="stocks")
    # print()

    download()
