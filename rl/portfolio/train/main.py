import mlflow
from mlflow_utils import init_mlflow, load_file, log_file
import itertools

import pandas as pd
from rl.portfolio.eval.envs.env import EvalEnv
from rl.portfolio.eval.envs.pre_process.pre_process import PreProcessor
from utils.paths import mlflow_dir
from rl.portfolio.train.envs.pre_process.order_daywise import order_daywise


def main():
    init_mlflow(mlflow_dir, "Tests")
    ticker = load_file(
        run_id="f384f58217114433875eda44495272ad",
        fn="evl_ticker.pkl",
        experiment="Eval_Stocks",
    )

    sorted_buys = order_daywise(ticker)
