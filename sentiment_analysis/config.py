from utils import Config
from datetime import datetime
import paths

general = Config(**dict(
    start=datetime(year=2021, month=1, day=28),
    end=datetime(year=2021, month=2, day=4),
    # path=paths.sentiment_data_path / "28-01-21 - 04-02-21_1",
    path=None,
))

gc = Config(**dict(
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
    integrity=True
))

submissions = Config(**dict(
    search_ticker_in_body=True,
    ticker_blacklist=["DD"],
    body_col="selftext",
    cols_in_vader_merge=["id", "num_comments", "score", "date", "pos", "compound", "neu", "neg", "date_mesz"],
))
