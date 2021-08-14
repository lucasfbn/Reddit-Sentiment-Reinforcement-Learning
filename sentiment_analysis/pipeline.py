import mlflow

from sentiment_analysis.tasks import *
from utils.mlflow_api import log_file
from utils.pipeline_utils import seq_map
from utils.util_funcs import update_check_key

mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("Sentiment")

params = {
    "gc_dump_fn": "gc_dump.csv",
    "report_fn": "report.csv",
    "start": None,
    "end": None,
    "check_duplicates": True,
    "fields_to_retrieve": ["author", "created_utc", "id", "num_comments", "score", "title", "selftext",
                           "subreddit"],
    "cols_to_check_if_removed": ["author", "selftext", "title"],
    "filter_too_frequent_authors": True,
    "author_blacklist": [],
    "max_submissions_per_author_per_day": 1,
    "cols_to_be_cleaned_from_non_alphanumeric": ["title"],
    "ticker_blacklist": [],
    "search_ticker_in_body": True,
    "valid_ticker_path": str(paths.all_ticker),
    "relevant_timespan_cols": ["ticker", "num_comments", "score",
                               "pos", "neu", "neg", "compound"]
}


def pipeline(**kwargs):
    global params
    params = update_check_key(params, kwargs)

    df = get_from_gc(params["start"], params["end"], params["check_duplicates"], params["fields_to_retrieve"]).run()
    log_file(df, params["gc_dump_fn"])
    df = filter_removed(df, params["cols_to_check_if_removed"]).run()
    df = add_temporal_informations(df).run()
    df = filter_authors(df,
                        params["filter_too_frequent_authors"], params["author_blacklist"],
                        params["max_submissions_per_author_per_day"]).run()
    df = delete_non_alphanumeric(df, params["cols_to_be_cleaned_from_non_alphanumeric"]).run()

    valid_ticker = load_valid_ticker(params["valid_ticker_path"]).run()
    df = get_submission_ticker(df, valid_ticker=valid_ticker,
                               ticker_blacklist=params["ticker_blacklist"],
                               search_ticker_in_body=params["search_ticker_in_body"]).run()

    df = filter_submissions_without_ticker(df).run()
    df = merge_ticker_to_a_single_column(df).run()
    df = analyze_sentiment(df).run()
    flattened_ticker_df = flatten_ticker_scores(df).run()

    timespans = retrieve_timespans(flattened_ticker_df, params["relevant_timespan_cols"]).run()
    timespans = seq_map(aggregate_submissions_per_timespan, timespans).run()
    df = summarize_timespans(timespans).run()
    log_file(df, params["report_fn"])
    mlflow.log_params(params=params)
    return df


def main(test_mode=False):
    with mlflow.start_run():
        pipeline(start=datetime(year=2021, month=5, day=1), end=datetime(year=2021, month=5, day=14))


if __name__ == "__main__":
    main(True)
