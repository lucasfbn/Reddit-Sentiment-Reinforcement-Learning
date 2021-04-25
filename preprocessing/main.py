import mlflow

import mlflow

import paths
import preprocessing.config as config
from preprocessing.preprocessing_utils.cleaner import Cleaner
from preprocessing.preprocessing_utils.merge_preprocessing import MergePreprocessing
from preprocessing.preprocessing_utils.preprocessor import Preprocessor
from preprocessing.preprocessing_utils.timeseries_generator import TimeseriesGeneratorWrapper
from utils import save_config, Config


def main():
    from_run_id = config.general.from_run_id
    from_run = mlflow.get_run(from_run_id)
    from_artifact_path = paths.artifact_path(from_run.info.artifact_uri)

    meta_data = Config(**dict(
        start=from_run.data.params["start"],
        end=from_run.data.params["end"],
        from_run_id=from_run_id
    ))

    # to_artifact_path = paths.artifact_path(mlflow.get_artifact_uri())
    to_artifact_path = paths.artifact_path("file:///C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/4/8509d54a001f4dcc9766594fc0a73c89/artifacts")


    Preprocessor.source_path = from_artifact_path
    Preprocessor.target_path = to_artifact_path

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
    # mhp.pipeline()

    c = Cleaner(
        keep_offset=config.cleaner.keep_offset,
        cols_to_be_dropped=config.cleaner.cols_to_be_dropped,
        use_price=config.cleaner.use_price,
        min_len=config.general.min_len
    )
    # c.pipeline()

    tsg = TimeseriesGeneratorWrapper(
        metadata_cols=config.timeseries_generator.metadata_cols,
        check_availability=config.timeseries_generator.check_availability,
        look_back=config.timeseries_generator.look_back,
        scale=config.timeseries_generator.scale,
        keep_unscaled=config.timeseries_generator.keep_unscaled
    )
    tsg.pipeline()

    save_config(
        configs=[meta_data, config.general, config.merge_preprocessing, config.cleaner, config.timeseries_generator])


if __name__ == "__main__":
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Datasets")
    mlflow.start_run()

    main()

    mlflow.end_run()
