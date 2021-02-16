import paths
from preprocess.preprocessor import Preprocessor
from preprocess.merge_hype_price import MergeHypePrice
from preprocess.cleaner import Cleaner
from preprocess.timeseries_generator import TimeseriesGenerator

Preprocessor.path = paths.data_paths[3]

mhp = MergeHypePrice(
    start_hour=22,
    start_min=0,
    market_symbols=[],
    min_len=7,
    start_offset=30,
    live=False
)
# mhp.pipeline()

c = Cleaner(
    min_len=7,
    keep_offset=7
)
c.pipeline()

tsg = TimeseriesGenerator(
    look_back=7
)
tsg.pipeline()
