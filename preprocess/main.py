import paths
from preprocess.preprocessor import Preprocessor
from preprocess.merge_hype_price import MergeHypePrice
from preprocess.cleaner import Cleaner
from preprocess.timeseries_generator import TimeseriesGenerator

Preprocessor.path = paths.data_paths[6]

mhp = MergeHypePrice(
    start_hour=22,
    start_min=0,
    market_symbols=[],
    min_len_hype=9,
    start_offset=30,
    live=False,
    limit=10
)
mhp.pipeline()
# mhp.save_settings(mhp)

# c = Cleaner(
#     min_len_hype_price=7,
#     keep_offset=7
# )
# c.pipeline()
#
# tsg = TimeseriesGenerator(
#     look_back=7
# )
# tsg.pipeline()