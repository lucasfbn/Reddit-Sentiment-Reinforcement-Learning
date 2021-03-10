import paths
from preprocessing.preprocessing_utils.cleaner import Cleaner
from preprocessing.preprocessing_utils.merge_hype_price import MergePreprocessing
from preprocessing.preprocessing_utils.preprocessor import Preprocessor
from preprocessing.preprocessing_utils.timeseries_generator import TimeseriesGeneratorCNN
from utils import tracker

Preprocessor.source_path = paths.sentiment_data_path / "28-01-21 - 04-02-21_0"
Preprocessor.target_path = paths.create_dir(paths.datasets_data_path)
# Preprocessor.target_path = paths.datasets_data_path / "_4"

tracker.add({"source_path": str(Preprocessor.source_path.name),
             "target_path": str(Preprocessor.target_path.name)}, "Main")

mhp = MergePreprocessing(
    start_hour=21,
    start_min=0,
    market_symbols=[],
    min_len_hype=1,
    start_offset=30,
    fill_gaps=False,
    scale_cols_daywise=False,
    live=False,
    limit=None
)
mhp.pipeline()

c = Cleaner(
    min_len_hype_price=1,
    keep_offset=7
)
c.pipeline()
#
tsg = TimeseriesGeneratorCNN(
    look_back=7,
    scale=True
)
tsg.pipeline()

tracker.new(kind="datasets")
