import paths
from preprocess.preprocessor import Preprocessor
from preprocess.merge_clean.merge_hype_price import MergeHypePrice
from preprocess.merge_clean.cleaner import Cleaner
from preprocess.timeseries_generator.timeseries_generator import TimeseriesGenerator

from utils import tracker

Preprocessor.source_path = paths.sentiment_data_path / "13-01-21 - 25-01-21_0"
# Preprocessor.target_path = paths.create_dir(paths.datasets_data_path)
Preprocessor.target_path = paths.datasets_data_path / "_0"

tracker.add({"source_path": str(Preprocessor.source_path.name),
             "target_path": str(Preprocessor.target_path.name)}, "Main")

mhp = MergeHypePrice(
    start_hour=21,
    start_min=0,
    market_symbols=[],
    min_len_hype=7,
    start_offset=30,
    fill_gaps=False,
    scale_cols_daywise=False,
    live=False,
    limit=None
)
# mhp.pipeline()
# mhp.save_settings(mhp)

c = Cleaner(
    min_len_hype_price=7,
    keep_offset=7
)
# c.pipeline()
#
tsg = TimeseriesGenerator(
    look_back=7
)
tsg.pipeline()

# tracker.new(kind="datasets")
