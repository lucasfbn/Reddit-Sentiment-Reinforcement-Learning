import paths
from preprocessing.preprocessing_utils.timeseries_generator import TimeseriesGeneratorCNN
from utils import Config

general = Config(**dict(
    min_len=6,
    source_path=paths.sentiment_data_path / "13-01-21 - 25-01-21_3",
    target_path=paths.create_dir(paths.datasets_data_path),
    # target_path=paths.datasets_data_path / "_1"
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
    kind=TimeseriesGeneratorCNN,
    look_back=7,
    metadata_cols=["price", "tradeable", "date", "available"],
    check_availability=False,
    scale=True,
    keep_unscaled=True
))
