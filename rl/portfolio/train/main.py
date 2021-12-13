import mlflow
from mlflow_utils import init_mlflow, load_file, log_file
import itertools

import pandas as pd
from rl.portfolio.eval.envs.env import EvalEnv
from rl.portfolio.eval.envs.pre_process.pre_process import PreProcessor
from utils.paths import mlflow_dir
from rl.portfolio.train.envs.pre_process.order_daywise import order_sequences_daywise
from rl.portfolio.train.envs.utils.merge_ticker import merge_ticker


def main(data_run_id, eval_run_id):
    init_mlflow(mlflow_dir, "Tests")

    data = load_file(run_id=data_run_id, fn="ticker.pkl", experiment="Datasets")
    evl = load_file(run_id=eval_run_id, fn="evl_ticker.pkl", experiment="Eval_Stocks")

    merged = merge_ticker(data, evl)
    sorted_buys = order_sequences_daywise(merged)


main(data_run_id="0643613545e44e75b8017b9973598fb4", eval_run_id="f384f58217114433875eda44495272ad")
