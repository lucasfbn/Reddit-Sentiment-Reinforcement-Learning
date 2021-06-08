import math
import multiprocessing
from datetime import datetime, timedelta

import pandas as pd
import mlflow

import sentiment_analysis.reddit_data.api.pushshift as pushshift
import sentiment_analysis.reddit_data.api.reddit as reddit
from sentiment_analysis.reddit_data.api.google_cloud import BigQueryDB
from sentiment_analysis.reddit_data.worker import workers
from utils.utils import log
from utils.mlflow_api import log_file
import paths

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


def historic_data_wrapper(start, end, freq=5):
    def date_range(start, end, freq):
        days_diff = (end - start).days
        intv = math.ceil(days_diff / freq)
        diff = (end - start) / intv
        for i in range(intv):
            yield (start + diff * i)
        yield end

    dt_rng = list(date_range(start, end, freq))

    for i in range(len(dt_rng) - 1):
        historic_data(dt_rng[i], dt_rng[i + 1])


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


def get(start, end, use_pushshift=False):
    user_agent = "RedditTrend"
    client_id = "XGPKdr-omVKy6A"
    client_secret = "tXPhgIcvevtWoJzwFQXYXgyL5bF9JQ"
    username = "qrzte"
    password = "ga5FwRaXcRyJ77e"

    log.info(f"Start: {str(start)}. End: {str(end)}.")

    pushshift_api = pushshift.BetaAPI()
    reddit_api = reddit.RedditAPI(user_agent=user_agent, client_id=client_id, client_secret=client_secret)

    submissions = pd.DataFrame()

    if use_pushshift and pushshift_api.available():
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


def detect_gaps():
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Gaps")
    mlflow.start_run()

    db = BigQueryDB()
    gaps = db.detect_gaps(save_json=False)
    log_file(gaps, "gaps.json")

    mlflow.end_run()


if __name__ == '__main__':
    # hourly_scrape(0, 0)
    # start = datetime(year=2020, month=12, day=13)
    # end = datetime(year=2021, month=1, day=27, hour=14)
    # get(start, end)
    # historic_data(start, end)
    # historic_data_wrapper(start, end, freq=3)

    detect_gaps()
