import wandb
from simplepipeline import (Pipeline, get_pipeline, par_map, seq_map,
                            seq_map_unpack, set_pipeline)

from preprocessing.tasks import *
from utils.util_funcs import update_check_key
from utils.wandb_utils import load_artefact, log_artefact

params = {
    "input_df": None,
    "start_hour": 21,
    "start_min": 0,
    "sentiment_data_columns": ["num_comments", "score", "pos", "neu", "neg", "compound",
                               "num_posts"],
    "price_data_columns": ["Open", "High", "Low", "Close", "Volume"],
    "ticker_with_false_data": ["PHIL", "USMJ", "MINE", "SPONF", "TIPS"],
    "additional_metric_columns": [],
    "price_column": "Close",
    "drop_unscaled_cols": True,
    "ticker_min_len": 2,
    "enable_live_behaviour": False,
    "include_available_days_only": True,
    "sequence_length": 14,
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

    set_pipeline(Pipeline("Pre-Processing"))

    df = add_time(params["input_df"]).run()
    df = shift_time(df, params["start_hour"], params["start_min"]).run()
    df = drop_columns(df, params["start_end_columns"]).run()

    ticker = grp_by_ticker(df).run()
    ticker = seq_map(aggregate_daywise, ticker).run()
    ticker = seq_map(drop_ticker_with_too_few_data, ticker, ticker_min_len=params["ticker_min_len"]).run()
    ticker = remove_excluded_ticker(ticker).run()
    ticker = seq_map(mark_ticker_with_false_stock_data, ticker,
                     false_data_ticker=params["ticker_with_false_data"]).run()
    ticker = remove_excluded_ticker(ticker).run()
    ticker = seq_map(sort_ticker_df_chronologically, ticker, by=params["main_date_col_param"]).run()
    ticker = seq_map(mark_trainable_days, ticker, ticker_min_len=params["ticker_min_len"]).run()
    ticker = par_map(add_price_data, ticker,
                     price_data_start_offset=params["sequence_length"] + 1,
                     enable_live_behaviour=params["enable_live_behaviour"]).run()
    # clean_price_data_cache().run()
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

    seq_map(assert_no_nan, ticker).run()
    ticker = seq_map(copy_unscaled_price, ticker).run()

    cols_to_be_scaled = params["price_data_columns"] + params["additional_metric_columns"]

    # Several different lists of columns scaled separately so we can keep track of the changes made to each individual
    # list of column names

    # Scale price data
    ticker, new_price_data_columns = seq_map_unpack(scale, ticker,
                                                    cols_to_be_scaled=params["price_data_columns"],
                                                    drop_unscaled_cols=params["drop_unscaled_cols"]).run()
    params["price_data_columns"] = new_price_data_columns[0]  # Because new_price_data_columns is a list

    # Scale additional metrics
    ticker, new_additional_metric_columns = seq_map_unpack(scale, ticker,
                                                           cols_to_be_scaled=params["additional_metric_columns"],
                                                           drop_unscaled_cols=params["drop_unscaled_cols"]).run()
    params["additional_metric_columns"] = new_additional_metric_columns[0]

    # Scale sentiment data
    ticker, new_sentiment_data_columns = seq_map_unpack(scale, ticker,
                                                        cols_to_be_scaled=params["sentiment_data_columns"],
                                                        drop_unscaled_cols=params["drop_unscaled_cols"]).run()
    params["sentiment_data_columns"] = new_sentiment_data_columns[0]

    ticker = par_map(make_sequences, ticker,
                     sequence_length=params["sequence_length"],
                     include_available_days_only=params["include_available_days_only"],
                     columns_to_be_excluded_from_sequences=params["columns_to_be_excluded_from_sequences"],
                     price_column="price_scaled",
                     which=params["sequence_to_be_generated"]).run()
    ticker = seq_map(delete_non_tradeable_sequences, ticker).run()
    ticker = seq_map(mark_short_sequences, ticker, min_sequence_len=params["min_sequence_len"]).run()
    ticker = remove_excluded_ticker(ticker).run()

    log_artefact(ticker, "dataset.pkl", type="Datasets")

    params.pop("input_df")
    wandb.log(params=params)
    wandb.log({"Executed tasks": get_pipeline().executed_tasks()})
    return ticker


def main():
    with wandb.init(project="Trendstuff", group="Datasets") as run:
        df = load_artefact(run, fn="dataset.csv", version=0, type="Sentiment_Analysis")
        pipeline(input_df=df)


if __name__ == "__main__":
    main()
