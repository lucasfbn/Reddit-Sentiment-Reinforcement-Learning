import paths
from utils import Config

general = Config(**dict(
    min_len=6,
    from_run_id="97ba9499d6b745229d2fdc0d6d80af78"
))

merge_preprocessing = Config(**dict(
    start_hour=21,
    start_min=0,
    market_symbols=[],
    start_offset=30,
    fill_gaps=False,
    scale_cols_daywise=False,
    cols_to_be_scaled_daywise=['num_comments', "score", 'pos', 'compound', 'neu', 'neg', 'n_posts'],
    live=False,
    limit=None
))

cleaner = Config(**dict(
    cols_to_be_dropped=["date_day", "open", "close", "high", "low", "adj_close", "volume", "date_weekday",
                        "start_timestamp", "end_timestamp"],
    use_price="close",
    keep_offset=7,
))

timeseries_generator = Config(**dict(
    kind="nn",
    look_back=7,
    metadata_cols=["price", "tradeable", "date", "available"],
    check_availability=False,
    scale=True,
    keep_unscaled=False
))
