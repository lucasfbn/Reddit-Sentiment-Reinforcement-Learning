import pandas as pd
from datetime import datetime, timedelta

from utils import dt_to_timestamp, log
import preprocess.sentiment_analysis.api.pushshift as pushshift
import preprocess.sentiment_analysis.api.reddit as reddit
from preprocess.sentiment_analysis.api.google_cloud import BigQueryDB

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

user_agent = "RedditTrend"
client_id = "XGPKdr-omVKy6A"
client_secret = "tXPhgIcvevtWoJzwFQXYXgyL5bF9JQ"
username = "qrzte"
password = "ga5FwRaXcRyJ77e"


def hourly_scrape(data, context):
    end = datetime.now()
    start = end - timedelta(hours=1)

    log.info(f"Start: {str(start)}. End: {str(end)}.")

    pushshift_api = pushshift.BetaAPI()
    reddit_api = reddit.RedditAPI(user_agent=user_agent, client_id=client_id, client_secret=client_secret,
                                  username=username, password=password)

    submissions = pd.DataFrame()

    if pushshift_api.available():
        log.info("Using pushshift beta api.")

        submission_ids = []

        log.info("\tRetrieving submission ids.")
        for subreddit in subreddits:
            log.info(f"\t\tProcessing {subreddit}")
            ids = pushshift_api.get_submission_ids(start=start, subreddit=subreddit, filter_removed=True)
            submission_ids.extend(ids)

        log.info("\tRetrieving submissions.")
        for subm_id in submission_ids:
            reddit_api.get_submissions_by_id(ids=submission_ids)
            submissions = submissions.append(reddit_api.to_df())

    else:

        log.info("Using native reddit api.")

        log.info("\tRetrieving submissions.")
        for subreddit in subreddits:
            log.info(f"\t\tProcessing {subreddit}")
            reddit_api.get_latest_submissions(start=start, subreddit=subreddit)
            submissions = submissions.append(reddit_api.to_df())

    db = BigQueryDB()
    db.upload(submissions, dataset="data", table="submissions")


if __name__ == '__main__':
    hourly_scrape(0, 0)
