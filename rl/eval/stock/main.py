import mlflow
from mlflow_utils import artifact_path, init_mlflow, load_file, log_file
from mlflow_utils import setup_logger
from stable_baselines3 import PPO

import utils.paths
from rl.eval.stock.eval_ticker import Eval
from rl.train.envs.env import EnvCNN, EnvCNNExtended
from rl.train.envs.sub_envs.trading import SimpleTradingEnvEvaluation


def main(dataset_run_id, model_run_id, model_fn):
    init_mlflow(utils.paths.mlflow_dir, "Training")
    setup_logger("INFO")
    with mlflow.start_run(run_id=model_run_id, nested=True):
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
        eval_env.eval_ticker()

        log_file(eval_env.all_tracker_dict, "eval_stockwise.json")
        log_file(eval_env.agg_metadata_df, "agg_metadata.csv")
        log_file(eval_env.agg_metadata_stats, "agg_metadata_stats.csv")
        mlflow.log_params(eval_env.agg_metadata_stats_flat)
        mlflow.log_params({"dataset_run_id": dataset_run_id, "model_run_id": model_run_id, "model_fn": model_fn})


main(
    dataset_run_id="c3925e0cbdcd4620b5cb909e1a629419",
    model_run_id="0e29194a80494ad1a0ad03ab3b0eb30b",
    model_fn="rl_model_3233150_steps.zip"
)
