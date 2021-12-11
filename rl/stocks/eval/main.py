import mlflow
import pandas as pd
from mlflow_utils import artifact_path, init_mlflow, load_file, log_file
from mlflow_utils import setup_logger
from stable_baselines3 import PPO

import utils.paths
from rl.stocks.eval.agg_stats import AggStats
from rl.stocks.eval.eval_ticker import Eval
from rl.stocks.train.envs.env import EnvCNNExtended
from rl.stocks.train.envs.sub_envs.trading import SimpleTradingEnvEvaluation


def main(dataset_run_id, model_run_id, model_fn):
    init_mlflow(utils.paths.mlflow_dir, "Eval_Stocks")
    setup_logger("INFO")
    with mlflow.start_run():
        data = load_file(run_id=dataset_run_id, fn="ticker.pkl", experiment="Datasets")

        training_env = EnvCNNExtended
        trading_env = SimpleTradingEnvEvaluation

        model_path = (
                artifact_path(run_id=model_run_id, experiment="Training")
                / "models"
                / model_fn
        )
        model = PPO.load(model_path)

        eval_env = Eval(data, model, training_env, trading_env)
        evl_ticker = eval_env.eval_ticker()

        agg_evl_df = pd.DataFrame([{"ticker": ticker.name} | ticker.evl.to_dict() for ticker in evl_ticker])
        agg_stats = AggStats(agg_evl_df)
        agg_stats_df, agg_stats_json = agg_stats.agg()

        log_file(evl_ticker, "evl_ticker.pkl")
        log_file(agg_evl_df, "agg_evl_df.csv")
        log_file(agg_stats_df, "agg_stats.csv")
        mlflow.log_params(agg_stats_json)
        mlflow.log_params({"dataset_run_id": dataset_run_id, "model_run_id": model_run_id, "model_fn": model_fn})


main(
    dataset_run_id="0643613545e44e75b8017b9973598fb4",
    model_run_id="d0c7a104aad64e25b267021078a14781",
    model_fn="rl_model_3233150_steps.zip"
)
