import paths
from preprocess.preprocessing_utils.preprocessor import Preprocessor
from preprocess.preprocessing_utils.merge_hype_price import MergePreprocessing
from preprocess.preprocessing_utils.cleaner import Cleaner
from preprocess.preprocessing_utils.timeseries_generator import TimeseriesGenerator

Preprocessor.target_path = paths.d_path(17)

mhp = MergePreprocessing(
    start_hour=21,
    start_min=0,
    market_symbols=[],
    min_len_hype=1,
    start_offset=30,
    fill_gaps=False,
    scale_cols_daywise=False,
    live=True,
    limit=None
)
# mhp.pipeline()

c = Cleaner(
    min_len_hype_price=1,
    keep_offset=7
)
# c.pipeline()

tsg = TimeseriesGenerator(
    look_back=7,
    live=True
)
tsg.pipeline()
