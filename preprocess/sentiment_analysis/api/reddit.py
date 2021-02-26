import praw
from utils import dt_to_timestamp, log, submission_schema
import pandas as pd


class API:

    def __init__(self, user_agent, client_id, client_secret, username, password):
        self.reddit = praw.Reddit(user_agent=user_agent,
                                  client_id=client_id,
                                  client_secret=client_secret,
                                  username=username,
                                  password=password)
        self.submissions = []

    def get_latest_submissions(self, start, subreddit, limit=10):
        sub = self.reddit.subreddit(subreddit)
        sub_submissions = sub.new(limit=limit)

        submissions = []

        for subm in sub_submissions:
            if subm.created_utc < start:
                break
            subm = Submission(subm)
            submissions.append(subm.extract_relevant_data())

        self.submissions = submissions

    def get_submissions_by_id(self, ids):

        # Add reddit prefixes for objects, see: https://www.reddit.com/dev/api/
        ids = [f"t3_{id}" for id in ids]

        submissions = []

        for subm in self.reddit.info(fullnames=ids):
            subm = Submission(subm)
            submissions.append(subm.extract_relevant_data())

        self.submissions = submissions

    def to_df(self):
        df = pd.DataFrame(self.submissions)
        df = df.astype(submission_schema)
        return df


class Submission:

    def __init__(self, submission):
        self.submission = submission

    def extract_relevant_data(self):
        temp = {}
        submission_dict = self.submission.__dict__
        for schema_key in submission_schema:
            if schema_key in submission_dict:
                if schema_key == "author":
                    temp[schema_key] = self.submission.author.name
                elif schema_key == "subreddit":
                    temp[schema_key] = self.submission.subreddit.display_name
                else:
                    temp[schema_key] = submission_dict[schema_key]
            else:
                temp[schema_key] = None
        return temp


if __name__ == '__main__':
    user_agent = "RedditTrend"
    client_id = "XGPKdr-omVKy6A"
    client_secret = "tXPhgIcvevtWoJzwFQXYXgyL5bF9JQ"
    username = "qrzte"
    password = "ga5FwRaXcRyJ77e"

    d = API(user_agent=user_agent, client_id=client_id, client_secret=client_secret,
            username=username, password=password)
    d.get_latest_submissions(start=1, subreddit="wallstreetbets")
    d.get_submissions_by_id(ids=["ls36lf", "lrlbh5"])
    df = d.to_df()
    print()
