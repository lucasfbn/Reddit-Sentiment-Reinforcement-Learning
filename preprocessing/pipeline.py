import os

os.environ['PREFECT__LOGGING__LEVEL'] = "INFO"

import mlflow
import prefect
from prefect import Flow, Parameter, unmapped
from prefect.engine.state import Success
from prefect.executors import LocalExecutor, LocalDaskExecutor

import paths
from preprocessing.tasks import *
from utils.util_tasks import mlflow_log_file, unpack_union_mapping, reduce_list
from utils.mlflow_api import load_file
from preprocessing.ticker import Ticker

mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("Tests")

with Flow("preprocessing") as flow:
    input_df = Parameter("input_df")
    start_hour = Parameter("start_hour", default=21)
    start_min = Parameter("start_min", default=0)

    sentiment_data_columns = Parameter("sentiment_data_columns",
                                       default=["num_comments", "score", "pos", "neu", "neg", "compound",
                                                "num_posts"])
    price_data_columns = Parameter("price_data_columns", default=["Open", "High", "Low", "Close", "Volume"])
    price_column = Parameter("price_column", default="Close")

    drop_unscaled_cols = Parameter("drop_unscaled_cols", default=True)
    ticker_min_len = Parameter("ticker_min_len", default=2)
    price_data_start_offset = Parameter("price_data_start_offset", default=10)
    enable_live_behaviour = Parameter("enable_live_behaviour", default=False)
    include_available_days_only = Parameter("include_available_days_only", default=True)
    sequence_length = Parameter("sequence_length", default=3)
    columns_to_be_excluded_from_sequences = Parameter("columns_to_be_excluded_from_sequences",
                                                      default=["available", "tradeable", date_day_col,
                                                               "sentiment_data_available"])

    df = add_time(input_df)
    df = shift_time(df, start_hour, start_min)
    df = drop_columns(df, Parameter("start_end_columns", default=["start", "end"]))

    df, sentiment_data_columns = scale_sentiment_data_daywise(df, sentiment_data_columns, drop_unscaled_cols)
    ticker = grp_by_ticker(df)
    ticker = drop_ticker_df_columns.map(ticker, unmapped(Parameter("ticker_column", default=["ticker"])))
    ticker = drop_ticker_with_too_few_data(ticker, ticker_min_len)
    ticker = sort_ticker_df_chronologically.map(ticker, unmapped(Parameter("date_shifted", date_shifted_col)))
    ticker = mark_trainable_days.map(ticker, unmapped(ticker_min_len))
    ticker = add_price_data.map(ticker, unmapped(price_data_start_offset), unmapped(enable_live_behaviour))
    ticker = remove_excluded_ticker(ticker)
    ticker = mark_sentiment_data_available_days.map(ticker, unmapped(sentiment_data_columns))
    ticker = drop_ticker_df_columns.map(ticker, unmapped(Parameter("adj_close_column_plus_merged",
                                                                   default=["Adj Close", "_merge"])))
    ticker = sort_ticker_df_chronologically.map(ticker, unmapped(Parameter("date_day", date_day_col)))
    ticker = backfill_availability.map(ticker)
    ticker = assign_price_col.map(ticker, unmapped(price_column))
    ticker = drop_ticker_df_columns.map(ticker, unmapped(price_column))
    price_data_columns = remove_old_price_col_from_price_data_columns(price_data_columns, price_column)
    ticker = mark_tradeable_days.map(ticker)
    ticker = forward_fill_price_data.map(ticker, unmapped(price_data_columns))
    ticker = mark_ticker_where_all_prices_are_nan.map(ticker)
    ticker = mark_ipo_ticker.map(ticker)
    ticker = remove_excluded_ticker(ticker)
    ticker = fill_missing_sentiment_data.map(ticker, unmapped(sentiment_data_columns))
    ticker = add_metric_rel_price_change.map(ticker)
    ticker = drop_ticker_df_columns.map(ticker, unmapped(Parameter("date_cols", default=[date_col,
                                                                                         date_shifted_col,
                                                                                         date_day_shifted_col])))
    _ = mlflow_log_file(ticker, "ticker.pkl")
    _ = assert_no_nan.map(ticker)
    ticker = copy_unscaled_price.map(ticker)

    # Cannot unpack directly, therefore we need to unpack manually
    temp_result = scale_price_data.map(ticker, unmapped(price_data_columns), unmapped(drop_unscaled_cols))
    ticker, price_data_columns = unpack_union_mapping(temp_result)
    price_data_columns = reduce_list(price_data_columns)

    ticker = make_sequences.map(ticker, unmapped(sequence_length),
                                unmapped(include_available_days_only),
                                unmapped(columns_to_be_excluded_from_sequences),
                                unmapped(Parameter("price_col", default="price_scaled")))
    ticker = mark_empty_sequences.map(ticker)
    ticker = remove_excluded_ticker(ticker)

    _ = mlflow_log_file(ticker, "ticker.pkl")


def main(test_mode=False):
    # flow.executor = LocalDaskExecutor(scheduler="processes", num_workers=8)
    flow.executor = LocalExecutor()

    if test_mode:
        retrieve_task = flow.get_tasks("drop_ticker_df_columns")[0]
        task_states = {retrieve_task: Success("test_state",
                                              result=load_file("c47516069f3b40dfb1c88f5407659c96", "tickera.pkl")[:50])}
        with mlflow.start_run():
            flow.run(task_states=task_states)
    else:
        df = load_file("c47516069f3b40dfb1c88f5407659c96", "report.csv")
        df = df.head(500)
        with mlflow.start_run():
            flow.run(dict(input_df=df))


if __name__ == "__main__":
    # flow.register("test")
    # flow.visualize()
    main(False)
