import paths
from preprocessing.preprocessing_utils.cleaner import Cleaner
from preprocessing.preprocessing_utils.merge_preprocessing import MergePreprocessing
from preprocessing.preprocessing_utils.preprocessor import Preprocessor
from preprocessing.preprocessing_utils.timeseries_generator import TimeseriesGeneratorCNN
from utils import tracker

Preprocessor.source_path = paths.sentiment_data_path / "28-01-21 - 04-02-21_0"
# Preprocessor.target_path = paths.create_dir(paths.datasets_data_path)
Preprocessor.target_path = paths.datasets_data_path / "_7"

tracker.add({"source_path": str(Preprocessor.source_path.name),
             "target_path": str(Preprocessor.target_path.name)}, "Main")

Preprocessor.min_len = 2

mhp = MergePreprocessing(
    start_hour=21,
    start_min=0,
    market_symbols=[],
    start_offset=30,
    fill_gaps=False,
    scale_cols_daywise=False,
    live=False,
    limit=None
)
mhp.pipeline()

c = Cleaner(
    keep_offset=7
)
# c.pipeline()
#
tsg = TimeseriesGeneratorCNN(
    look_back=7,
    scale=True
)
# tsg.pipeline()

tracker.new(kind="datasets")
