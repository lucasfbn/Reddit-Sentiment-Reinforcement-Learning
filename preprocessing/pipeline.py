import mlflow

import paths
from preprocessing.tasks import *
from utils.mlflow_api import load_file, log_file, init_mlflow
from utils.pipeline_utils import initialize, par_map, seq_map
from utils.util_funcs import update_check_key
from utils.logger import setup_logger

initialize()

params = {
    "input_df": None,
    "start_hour": 21,
    "start_min": 0,
    "sentiment_data_columns": ["num_comments", "score", "pos", "neu", "neg", "compound",
                               "num_posts"],
    "price_data_columns": ["Open", "High", "Low", "Close", "Volume"],
    "additional_metric_columns": [],
    "price_column": "Close",
    "drop_unscaled_cols": True,
    "ticker_min_len": 6,
    "price_data_start_offset": 10,
    "enable_live_behaviour": False,
    "include_available_days_only": True,
    "sequence_length": 7,
    "columns_to_be_excluded_from_sequences": ["available", "tradeable", main_date_col,
                                              "sentiment_data_available", "price_raw", "Open_scaled",
                                              "High_scaled", "Low_scaled"],
    "sequence_to_be_generated": "arr",
    "main_date_col_param": main_date_col,
    "start_end_columns": ["start", "end"],
    "adj_close_column_plus_merged": ["Adj_Close", "_merge"],
    "min_sequence_len": 2
}


def pipeline(**kwargs):
    global params
    params = update_check_key(params, kwargs)

    df = add_time(params["input_df"]).run()
    df = shift_time(df, params["start_hour"], params["start_min"]).run()
    df = drop_columns(df, params["start_end_columns"]).run()

    df, params["sentiment_data_columns"] = scale_sentiment_data_daywise(df, params["sentiment_data_columns"],
                                                                        params["drop_unscaled_cols"]).run()
    ticker = grp_by_ticker(df).run()
    ticker = seq_map(aggregate_daywise, ticker).run()
    ticker = seq_map(drop_ticker_with_too_few_data, ticker, ticker_min_len=params["ticker_min_len"]).run()
    ticker = remove_excluded_ticker(ticker).run()
    ticker = seq_map(sort_ticker_df_chronologically, ticker, by=params["main_date_col_param"]).run()
    ticker = seq_map(mark_trainable_days, ticker, ticker_min_len=params["ticker_min_len"]).run()
    ticker = par_map(add_price_data, ticker,
                     price_data_start_offset=params["price_data_start_offset"],
                     enable_live_behaviour=params["enable_live_behaviour"]).run()
    clean_price_data_cache().run()
    ticker = remove_excluded_ticker(ticker).run()
    ticker = seq_map(mark_sentiment_data_available_days, ticker,
                     sentiment_data_columns=params["sentiment_data_columns"]).run()
    ticker = seq_map(drop_ticker_df_columns, ticker, columns_to_be_dropped=params["adj_close_column_plus_merged"]).run()
    ticker = seq_map(sort_ticker_df_chronologically, ticker, by=params["main_date_col_param"]).run()
    ticker = seq_map(backfill_availability, ticker).run()
    ticker = seq_map(assign_price_col, ticker, price_col=params["price_column"]).run()
    ticker = seq_map(drop_ticker_df_columns, ticker, columns_to_be_dropped=params["price_column"]).run()
    params["price_data_columns"] = remove_old_price_col_from_price_data_columns(
        price_data_columns=params["price_data_columns"],
        price_column=params["price_column"]).run()
    ticker = seq_map(mark_tradeable_days, ticker).run()
    ticker = seq_map(forward_fill_price_data, ticker, price_data_cols=params["price_data_columns"]).run()
    ticker = seq_map(mark_ticker_where_all_prices_are_nan, ticker).run()
    ticker = seq_map(mark_ipo_ticker, ticker).run()
    ticker = remove_excluded_ticker(ticker).run()
    ticker = seq_map(fill_missing_sentiment_data, ticker, sentiment_data_columns=params["sentiment_data_columns"]).run()

    ticker = seq_map(add_metric_rel_price_change, ticker, metric_name="price_rel_change").run()
    params["additional_metric_columns"].append("price_rel_change")

    _ = log_file(ticker, "ticker.pkl")
    seq_map(assert_no_nan, ticker).run()
    ticker = seq_map(copy_unscaled_price, ticker).run()

    cols_to_be_scaled = params["price_data_columns"] + params["additional_metric_columns"]
    # map(list, ...) splits the list of tuples to two lists, see tests of pipeline_utils
    ticker, price_data_columns = map(list, zip(*seq_map(scale_price_data, ticker,
                                                        cols_to_be_scaled=cols_to_be_scaled,
                                                        drop_unscaled_cols=params["drop_unscaled_cols"]).run()))
    # params["price_data_columns"] = price_data_columns[0]

    ticker = par_map(make_sequences, ticker,
                     sequence_length=params["sequence_length"],
                     include_available_days_only=params["include_available_days_only"],
                     columns_to_be_excluded_from_sequences=params["columns_to_be_excluded_from_sequences"],
                     price_column="price_scaled",
                     which=params["sequence_to_be_generated"]).run()
    ticker = seq_map(mark_short_sequences, ticker, min_sequence_len=params["min_sequence_len"]).run()
    ticker = remove_excluded_ticker(ticker).run()

    log_file(ticker, "ticker.pkl")

    params.pop("input_df")
    mlflow.log_params(params=params)
    return ticker


def main():
    init_mlflow("Datasets")
    with mlflow.start_run():
        df = load_file(run_id="dec6cb437cbc455fa871d1ac5b9300c8", fn="report.csv", experiment="Sentiment")
        setup_logger("DEBUG")
        pipeline(input_df=df)


if __name__ == "__main__":
    main()
