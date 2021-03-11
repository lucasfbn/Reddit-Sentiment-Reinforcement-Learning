import preprocessing.config as config
from preprocessing.preprocessing_utils.cleaner import Cleaner
from preprocessing.preprocessing_utils.merge_preprocessing import MergePreprocessing
from preprocessing.preprocessing_utils.preprocessor import Preprocessor
from utils import save_config

Preprocessor.source_path = config.general.source_path
Preprocessor.target_path = config.general.target_path

mhp = MergePreprocessing(
    start_hour=config.merge_preprocessing.start_hour,
    start_min=config.merge_preprocessing.start_min,
    min_len=config.general.min_len,
    market_symbols=config.merge_preprocessing.market_symbols,
    start_offset=config.merge_preprocessing.start_offset,
    fill_gaps=config.merge_preprocessing.fill_gaps,
    scale_cols_daywise=config.merge_preprocessing.scale_cols_daywise,
    cols_to_be_scaled_daywise=config.merge_preprocessing.cols_to_be_scaled_daywise,
    live=config.merge_preprocessing.live,
    limit=config.merge_preprocessing.limit,
)
mhp.pipeline()

c = Cleaner(
    keep_offset=config.cleaner.keep_offset,
    cols_to_be_dropped=config.cleaner.cols_to_be_dropped,
    use_price=config.cleaner.use_price,
    min_len=config.general.min_len
)
c.pipeline()

tsg = config.timeseries_generator.kind
tsg = tsg(
    metadata_cols=config.timeseries_generator.metadata_cols,
    check_availability=config.timeseries_generator.check_availability,
    look_back=config.timeseries_generator.look_back,
    scale=config.timeseries_generator.scale
)

tsg.pipeline()

save_config(configs=[config.general, config.merge_preprocessing, config.cleaner, config.timeseries_generator],
            kind="datasets")
