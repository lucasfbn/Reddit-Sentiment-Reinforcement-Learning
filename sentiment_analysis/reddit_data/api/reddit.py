import praw
from utils import dt_to_timestamp
from sentiment_analysis.reddit_data.api.api import API


class RedditAPI(API):

    def __init__(self, user_agent, client_id, client_secret):
        self.reddit = praw.Reddit(user_agent=user_agent,
                                  client_id=client_id,
                                  client_secret=client_secret)
        self.submissions = []

    def extract_relevant_data(self, submission):
        temp = {}
        submission_dict = submission.__dict__
        for schema_key in self.submission_schema:
            if schema_key in submission_dict:
                if schema_key == "author":
                    try:
                        author_name = submission.author.name
                    except:
                        author_name = None
                    temp[schema_key] = author_name
                elif schema_key == "subreddit":
                    temp[schema_key] = submission.subreddit.display_name
                else:
                    temp[schema_key] = submission_dict[schema_key]
            else:
                temp[schema_key] = None
        return temp

    def get_latest_submissions(self, start, subreddit, limit):

        start = dt_to_timestamp(start)

        sub = self.reddit.subreddit(subreddit)
        sub_submissions = sub.new(limit=limit)

        submissions = []

        for subm in sub_submissions:
            if subm.created_utc < start:
                break
            submissions.append(self.extract_relevant_data(subm))

        self.submissions = submissions

    def get_submissions_by_id(self, ids):

        # Add reddit prefixes for objects, see: https://www.reddit.com/dev/api/
        ids = [f"t3_{id}" for id in ids]

        submissions = []

        for subm in self.reddit.info(fullnames=ids):
            submissions.append(self.extract_relevant_data(subm))

        self.submissions = submissions


if __name__ == '__main__':
    user_agent = "RedditTrend"
    client_id = "XGPKdr-omVKy6A"
    client_secret = "tXPhgIcvevtWoJzwFQXYXgyL5bF9JQ"
    username = "qrzte"
    password = "ga5FwRaXcRyJ77e"
    from datetime import datetime

    start = datetime(year=2021, month=2, day=23, hour=21)
    d = RedditAPI(user_agent=user_agent, client_id=client_id, client_secret=client_secret,
                  username=username, password=password)
    # d.get_latest_submissions(start=start, subreddit="wallstreetbets")
    d.get_submissions_by_id(ids=["ls36lf", "lrlbh5"])
    df = d.to_df()
    print()
