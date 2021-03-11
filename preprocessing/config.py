import paths
from preprocessing.preprocessing_utils.timeseries_generator import TimeseriesGeneratorCNN

general = dict(
    min_len=3,
    source_path=paths.sentiment_data_path / "13-01-21 - 25-01-21_0",
    # target_path=paths.create_dir(paths.datasets_data_path),
    target_path=paths.datasets_data_path / "_6"
)

merge_preprocessing = dict(
    start_hour=21,
    start_min=0,
    market_symbols=[],
    start_offset=30,
    fill_gaps=False,
    scale_cols_daywise=False,
    live=False,
    limit=None
)

cleaner = dict(
    keep_offset=7
)

timeseries_generator = dict(
    kind=TimeseriesGeneratorCNN,
    look_back=7,
    scale=True
)
