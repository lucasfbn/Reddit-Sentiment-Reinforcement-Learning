from datetime import datetime

from preprocess.sentiment_analysis.db.db_handler import DB
from preprocess.sentiment_analysis.download.from_pushshift.graber import Graber
from preprocess.sentiment_analysis.utils.utils import *


def download():
    g = Graber()
    database = DB()

    start = datetime(year=2021, month=2, day=15)
    end = datetime(year=2021, month=2, day=24)

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
        log.info(f"Processing {subreddit}. {i + 1}/{len(subreddits)}")
        try:
            data = g.get_submissions(start, end, subreddit=subreddit)
            database.up(data, dataset="data", table="submissions")
        except Exception as e:
            log.critical(f"Processing {subreddit} failed. Error: {e}")


if __name__ == '__main__':
    download()
