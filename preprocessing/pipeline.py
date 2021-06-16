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
from preprocessing.ticker import Ticker

mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("Tests")

with Flow("preprocessing") as flow:
    input_df = Parameter("input_df")
    start_hour = Parameter("start_hour", default=21)
    start_min = Parameter("start_min", default=0)

    sentiment_data_columns = Parameter("sentiment_data_columns",
                                       default=["num_comments", "score", "pos", "neu", "neg", "compound",
                                                "n_posts"])
    price_data_columns = Parameter("price_data_columns", default=["Open", "High", "Low", "Close", "Volume"])
    price_column = Parameter("price_column", default="Close")

    drop_unscaled_cols = Parameter("drop_unscaled_cols", default=True)
    ticker_min_len = Parameter("ticker_min_len", default=2)
    price_data_start_offset = Parameter("price_data_start_offset", default=10)
    enable_live_behaviour = Parameter("enable_live_behaviour", default=False)
    include_available_days_only = Parameter("include_available_days_only", default=True)
    sequence_length = Parameter("sequence_length", default=3)
    columns_to_be_excluded_from_sequences = Parameter("columns_to_be_excluded_from_sequences", default=["available",
                                                                                                        "tradeable"])

    df = add_time(input_df)
    df = shift_time(df, start_hour, start_min)
    df = drop_columns(df, Parameter("start_end_columns", default=["start", "start_timestamp", "end", "end_timestamp"]))
    df = drop_columns(df, Parameter("run_id_subreddit_columns", default=["run_id", "subreddit"]))

    df, sentiment_data_columns = scale_sentiment_data_daywise(df, sentiment_data_columns, drop_unscaled_cols)
    ticker = grp_by_ticker(df)
    ticker = drop_ticker_df_columns.map(ticker, unmapped(Parameter("ticker_column", default=["ticker"])))
    ticker = drop_ticker_with_too_few_data(ticker, ticker_min_len)
    ticker = sort_ticker_df_chronologically.map(ticker)
    ticker = mark_trainable_days.map(ticker, unmapped(ticker_min_len))
    ticker = add_price_data.map(ticker, unmapped(price_data_start_offset), unmapped(enable_live_behaviour))
    ticker = remove_excluded_ticker(ticker)
    ticker = drop_ticker_df_columns.map(ticker, unmapped(Parameter("adj_close_column", default=["Adj Close"])))
    ticker = sort_ticker_df_chronologically.map(ticker)
    ticker = backfill_availability.map(ticker)
    ticker = assign_price_col.map(ticker, unmapped(price_column))
    ticker = drop_ticker_df_columns.map(ticker, unmapped(price_column))
    price_data_columns = remove_old_price_col_from_price_data_columns(price_data_columns, price_column)
    ticker = mark_tradeable_days.map(ticker)

    ticker = drop_ticker_df_columns.map(ticker, unmapped(Parameter("date_cols", default=[date_col, date_day_col,
                                                                                         date_shifted_col,
                                                                                         date_day_shifted_col])))
    ticker = forward_fill_price.map(ticker)
    ticker = mark_ticker_where_all_prices_are_nan.map(ticker)
    ticker = mark_ipo_ticker.map(ticker)
    ticker = remove_excluded_ticker(ticker)
    ticker = fill_missing_sentiment_data.map(ticker, unmapped(sentiment_data_columns))
    _ = assert_no_nan.map(ticker)
    ticker = add_metric_rel_price_change.map(ticker)

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
    flow.executor = LocalDaskExecutor(scheduler="processes", num_workers=8)
    # flow.executor = LocalExecutor()

    if test_mode:
        import pickle as pkl
        with open(
                "C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/8/a974b4d81a434ea793e93e255ca18b19/artifacts/ticker.pkl",
                "rb") as f:
            file = pkl.load(f)
        task = flow.get_tasks("temp")[0]
        task_states = {task: Success("test_mode", result=file)}
        with mlflow.start_run():
            flow.run(task_states=task_states)
    else:
        df = pd.read_csv(
            "C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/3/39aaaaa5c33741218844a315f229093d/artifacts/report.csv",
            sep=";")
        df = df.head(1000)
        with mlflow.start_run():
            flow.run(dict(input_df=df))


if __name__ == "__main__":
    # flow.register("test")
    # flow.visualize()
    main(False)
