import praw


class Downloader:

    def __init__(self, user_agent, client_id, client_secret, username, password, subreddit, begin):
        self.api = praw.Reddit(user_agent=user_agent,
                               client_id=client_id,
                               client_secret=client_secret,
                               username=username,
                               password=password)
        self.subreddit = subreddit
        self.begin = begin

    def download(self):
        sub = self.api.subreddit(self.subreddit)
        submissions = sub.new()

        for subm in submissions:
            print()


class Submission:

    def __init__(self, submission_data):
        self.submission_data = submission_data

    def _extract_relevant(self):
        pass

    def to_df(self):
        pass


if __name__ == '__main__':
    user_agent = "RedditTrend"
    client_id = "XGPKdr-omVKy6A"
    client_secret = "tXPhgIcvevtWoJzwFQXYXgyL5bF9JQ"
    username = "qrzte"
    password = "ga5FwRaXcRyJ77e"

    d = Downloader(user_agent=user_agent, client_id=client_id, client_secret=client_secret,
                   username=username, password=password, subreddit="wallstreetbets", begin=1)
    d.download()
