import preprocessing.config as config
from preprocessing.preprocessing_utils.cleaner import Cleaner
from preprocessing.preprocessing_utils.merge_preprocessing import MergePreprocessing
from preprocessing.preprocessing_utils.preprocessor import Preprocessor
from utils import save_config

Preprocessor.min_len = config.general.min_len
Preprocessor.source_path = config.general.source_path
Preprocessor.target_path = config.general.target_path

mhp = MergePreprocessing(
    start_hour=config.merge_preprocessing.start_hour,
    start_min=config.merge_preprocessing.start_min,
    market_symbols=config.merge_preprocessing.market_symbols,
    start_offset=config.merge_preprocessing.start_offset,
    fill_gaps=config.merge_preprocessing.fill_gaps,
    scale_cols_daywise=config.merge_preprocessing.scale_cols_daywise,
    live=config.merge_preprocessing.live,
    limit=config.merge_preprocessing.limit,
)
mhp.pipeline()

c = Cleaner(
    keep_offset=config.cleaner.keep_offset
)
c.pipeline()

tsg = config.timeseries_generator.kind
tsg = tsg(
    look_back=config.timeseries_generator.look_back,
    scale=config.timeseries_generator.scale
)

tsg.pipeline()

save_config(configs=[config.general, config.merge_preprocessing, config.cleaner, config.timeseries_generator],
            kind="dataset")
