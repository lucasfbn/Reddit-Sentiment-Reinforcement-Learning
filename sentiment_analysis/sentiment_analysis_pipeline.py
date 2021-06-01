from prefect import Flow, Parameter
from prefect.executors import LocalDaskExecutor

from sentiment_analysis.logic.analyzer import *

gc_dump_fn = "gc_dump.csv"
report_fn = "report.csv"

with Flow("sentiment_analysis") as flow:
    """
    This pipeline downloads the data between a given start and end date from google big query and stores it in the
    current mlflow run. It then proceeds to apply some basic preprocessing.
    """

    start = Parameter("start")
    end = Parameter("end")
    check_duplicates = Parameter("check_duplicates", default=True)
    fields_to_retrieve = Parameter("fields_to_retrieve",
                                   default=["author", "created_utc", "id", "num_comments", "score", "title", "selftext",
                                            "subreddit"])
    cols_to_check_if_removed = Parameter("cols_to_check_if_removed", default=["author", "selftext", "title"])
    filter_too_frequent_authors = Parameter("filter_too_frequent_authors", default=True)
    author_blacklist = Parameter("author_blacklist", default=[])
    max_submissions_per_author_per_day = Parameter("max_submissions_per_author_per_day", default=1)
    cols_to_be_cleaned_from_non_alphanumeric = Parameter("cols_to_be_cleaned_from_non_alphanumeric", default=["title"])

    ticker_blacklist = Parameter("ticker_blacklist", default=[])
    search_ticker_in_body = Parameter("search_ticker_in_body", default=True)
    valid_ticker_path = Parameter("valid_ticker_path", default=str(paths.all_ticker))

    df = get_from_gc(start, end, check_duplicates, fields_to_retrieve)
    _ = mlflow_log_file(df, gc_dump_fn)
    df = filter_removed(df, cols_to_check_if_removed)
    df = add_temporal_informations(df)
    df = filter_authors(df, filter_too_frequent_authors, author_blacklist, max_submissions_per_author_per_day)
    df = delete_non_alphanumeric(df, cols_to_be_cleaned_from_non_alphanumeric)

    valid_ticker = load_valid_ticker(valid_ticker_path)
    df = get_submission_ticker(df, valid_ticker=valid_ticker,
                               ticker_blacklist=ticker_blacklist,
                               search_ticker_in_body=search_ticker_in_body)

    df = filter_submissions_without_ticker(df)
    df = merge_ticker_to_a_single_column(df)
    df = analyze_sentiment(df)
    flattened_ticker_df = flatten_ticker_scores(df)

    relevant_timespan_cols = Parameter("relevant_timespan_cols", default=["ticker", "num_comments", "score",
                                                                          "pos", "neu", "neg", "compound"])

    timespans = retrieve_timespans(flattened_ticker_df, relevant_timespan_cols)
    df = summarize_timespans(timespans)
    _ = mlflow_log_file(df, report_fn)

# flow.executor = LocalDaskExecutor(scheduler="processes", num_workers=8)

if __name__ == "__main__":
    import mlflow

    # flow.register("test")
    # flow.visualize()
    # flow.run(dict(start=datetime(year=2021, month=2, day=18),
    #               end=datetime(year=2021, month=2, day=20)))

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Tests")

    with mlflow.start_run():
        flow.run(dict(start=datetime(year=2021, month=2, day=18), end=datetime(year=2021, month=2, day=20)))
