import pandas as pd
from datetime import datetime, timedelta
import multiprocessing
import math
from utils import log
import preprocess.sentiment_analysis.reddit_data.api.pushshift as pushshift
import preprocess.sentiment_analysis.reddit_data.api.reddit as reddit
from preprocess.sentiment_analysis.reddit_data.worker import workers
from preprocess.sentiment_analysis.reddit_data.api.google_cloud import BigQueryDB

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


def _work(procnum, submission_id_chunk, worker, return_dict):
    worker = reddit.RedditAPI(user_agent=worker["user_agent"], client_id=worker["client_id"],
                              client_secret=worker["client_secret"])
    worker.get_submissions_by_id(submission_id_chunk)
    submissions = worker.to_df()
    return_dict[procnum] = submissions


def historic_data(start, end):
    pushshift_api = pushshift.MainApi()

    submissions_ids = []

    for subreddit in subreddits:
        log.info(f"Processing {subreddit}")
        submissions_ids.extend(pushshift_api.get_submission_ids(start, end, subreddit, True))

    n_worker = len(workers)

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    elements_per_chunk = math.ceil(len(submissions_ids) / n_worker)
    submissions_ids_splitted = list(chunks(submissions_ids, elements_per_chunk))

    jobs = []

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for i, (submission_id_chunk, worker) in enumerate(zip(submissions_ids_splitted, workers)):
        p = multiprocessing.Process(target=_work, args=(i, submission_id_chunk, worker, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    parent_df = pd.DataFrame()

    for val in return_dict.values():
        parent_df = parent_df.append(val)

    db = BigQueryDB()
    db.upload(parent_df, dataset="data", table="submissions")


def hourly_scrape(data, context):
    end = datetime.now()
    start = end - timedelta(hours=1)
    get(start, end)


def get(start, end):
    user_agent = "RedditTrend"
    client_id = "XGPKdr-omVKy6A"
    client_secret = "tXPhgIcvevtWoJzwFQXYXgyL5bF9JQ"
    username = "qrzte"
    password = "ga5FwRaXcRyJ77e"

    log.info(f"Start: {str(start)}. End: {str(end)}.")

    pushshift_api = pushshift.BetaAPI()
    reddit_api = reddit.RedditAPI(user_agent=user_agent, client_id=client_id, client_secret=client_secret)

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
        reddit_api.get_submissions_by_id(ids=submission_ids)
        submissions = submissions.append(reddit_api.to_df())

    else:

        log.info("Using native reddit api.")

        log.info("\tRetrieving submissions.")
        for subreddit in subreddits:
            log.info(f"\t\tProcessing {subreddit}")
            reddit_api.get_latest_submissions(start=start, subreddit=subreddit, limit=1000)
            submissions = submissions.append(reddit_api.to_df())

    db = BigQueryDB()
    db.upload(submissions, dataset="data", table="submissions")


if __name__ == '__main__':
    # hourly_scrape(0, 0)
    start = datetime(year=2021, month=2, day=5, hour=0)
    end = datetime(year=2021, month=2, day=7, hour=23)
    # get(start, end)
    historic_data(start, end)
