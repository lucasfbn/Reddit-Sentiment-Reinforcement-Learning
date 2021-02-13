import pickle as pkl
import pandas as pd
import preprocess.merge_hype_yahoo
import preprocess.clean_data
import preprocess.prepare_timeseries

import paths


def pipeline(input_path, output_path):
    df = pd.read_csv(input_path, sep=",")
    data = preprocess.merge_hype_yahoo.pipeline(df)
    data = preprocess.clean_data.pipeline(data)
    data = preprocess.prepare_timeseries.pipeline(data)

    with open(output_path, "wb") as f:
        pkl.dump(data, f)


path = paths.live_path / "13_02_2021"
pipeline(path / "report.csv", path / "data.pkl")
