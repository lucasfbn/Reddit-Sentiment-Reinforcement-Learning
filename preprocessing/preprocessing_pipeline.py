from prefect import Flow, Parameter, unmapped
from prefect.executors import LocalDaskExecutor
from utils.util_tasks import mlflow_log_file

from preprocessing.logic.logic import *

with Flow("preprocessing") as flow:
    input_df = Parameter("input_df")
    start_hour = Parameter("start_hour", default=21)
    start_min = Parameter("start_min", default=0)
    cols_to_be_scaled = Parameter("cols_to_be_scaled",
                                  default=["num_comments", "score", "pos", "neu", "neg", "compound",
                                           "n_posts"])
    drop_scaled_cols = Parameter("drop_scaled_cols", default=True)
    ticker_min_len = Parameter("ticker_min_len", default=2)
    price_data_start_offset = Parameter("price_data_start_offset", default=10)
    enable_live_behaviour = Parameter("enable_live_behaviour", default=False)

    df = add_time(input_df)
    df = shift_time(df, start_hour, start_min)
    min_time, max_time = get_min_max_time(df)
    _ = mlflow_log_file(df, "df.csv")
    df = scale_daywise(df, cols_to_be_scaled, drop_scaled_cols)
    ticker = grp_by_ticker(df)
    ticker = drop_ticker_with_too_few_data(ticker, ticker_min_len)
    ticker = sort_ticker_df_chronologically.map(ticker)
    ticker = mark_trainable_days.map(ticker, unmapped(ticker_min_len))
    ticker = add_price_data.map(ticker, unmapped(price_data_start_offset), unmapped(enable_live_behaviour))
    ticker = remove_excluded_ticker(ticker)
    ticker = sort_ticker_df_chronologically.map(ticker)
    ticker = backfill_availability.map(ticker)
    _ = mlflow_log_file(ticker, "test.pkl")

# flow.executor = LocalDaskExecutor(scheduler="processes", num_workers=8)

if __name__ == "__main__":
    import mlflow
    import paths

    # flow.register("test")
    # flow.visualize()

    df = pd.read_csv(
        "C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/3/39aaaaa5c33741218844a315f229093d/artifacts/report.csv",
        sep=";")

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Tests")

    with mlflow.start_run():
        flow.run(dict(input_df=df))
