import mlflow
from mlflow_utils import artifact_path, init_mlflow, load_file, log_file
from mlflow_utils import setup_logger
from stable_baselines3 import PPO

import utils.paths
from rl.eval.stock.eval_ticker import Eval
from rl.train.envs.env import EnvCNN
from rl.train.envs.sub_envs.trading import SimpleTradingEnvEvaluation


def main():
    init_mlflow(utils.paths.mlflow_dir, "Tests")
    setup_logger("INFO")
    with mlflow.start_run():
        data = load_file(run_id="5896df8aa22c41a3ade34d747bc9ed9a", fn="ticker.pkl", experiment="Datasets")

        training_env = EnvCNN
        trading_env = SimpleTradingEnvEvaluation

        model_path = (
                artifact_path(run_id="658bd49019894ffc99f1c800070b0be4", experiment="Tests")
                / "models"
                / "rl_model_680052_steps.zip"
        )
        model = PPO.load(model_path)

        eval_env = Eval(data, model, training_env, trading_env)
        eval_env.eval_ticker()

        log_file(eval_env.all_tracker_dict, "eval_stockwise.json")
        log_file(eval_env.agg_metadata_df, "agg_metadata.csv")
        log_file(eval_env.agg_metadata_stats, "agg_metadata_stats.csv")
        mlflow.log_params(eval_env.agg_metadata_stats_flat)


main()
