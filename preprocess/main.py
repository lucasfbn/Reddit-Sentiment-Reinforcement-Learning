import paths
from preprocess.preprocessor import Preprocessor
from preprocess.merge_hype_price import MergeHypePrice
from preprocess.cleaner import Cleaner
from preprocess.timeseries_generator import TimeseriesGenerator

Preprocessor.source_path = paths.sentiment_data_path / "13-01-21 - 25-01-21_0"
Preprocessor.target_path = paths.d_path(18)

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
mhp.pipeline()
# mhp.save_settings(mhp)

c = Cleaner(
    min_len_hype_price=7,
    keep_offset=7
)
c.pipeline()
#
tsg = TimeseriesGenerator(
    look_back=7
)
tsg.pipeline()
