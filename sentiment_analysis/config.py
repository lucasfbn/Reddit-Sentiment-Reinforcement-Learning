from datetime import datetime

import paths
from utils import Config

general = Config(**dict(
    start=datetime(year=2021, month=2, day=18),
    end=datetime(year=2021, month=4, day=25),
))

gc = Config(**dict(
    check_duplicates=True,
    fields=["author", "created_utc", "id", "num_comments", "score", "title", "selftext", "subreddit"]
))

preprocess = Config(**dict(
    author_blacklist=[],
    cols_to_check_if_removed=["author", "selftext", "title"],
    cols_to_be_cleaned=["title"],
    max_subm_p_author_p_day=1,
    filter_authors=True
))

check = Config(**dict(
    integrity=False
))

submissions = Config(**dict(
    search_ticker_in_body=False,
    ticker_blacklist=["DD"],
    body_col="selftext",
    cols_in_vader_merge=["id", "num_comments", "score", "date", "pos", "compound", "neu", "neg", "date_mesz"],
))
