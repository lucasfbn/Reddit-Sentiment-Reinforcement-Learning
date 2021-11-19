import mlflow
from mlflow_utils import artifact_path, init_mlflow, load_file, log_file
from stable_baselines3 import PPO

from rl._eval.envs.env import EvalEnv
from rl._eval.envs.pre_process.pre_process import PreProcessor
from rl.train.envs.env import EnvCNN, EnvCNNExtended
from utils.paths import mlflow_dir


def main():
    init_mlflow(mlflow_dir, "Tests")
    ticker = load_file(
        run_id="5e792abfb8de4869a4d4830fd3f61716",
        fn="ticker.pkl",
        experiment="Datasets",
    )
    pre_processor = PreProcessor()
    training_env = EnvCNN(ticker)
    model_path = (
            artifact_path(run_id="658bd49019894ffc99f1c800070b0be4", experiment="Tests")
            / "models"
            / "rl_model_680052_steps.zip"
    )
    model = PPO.load(model_path, training_env)

    eval_env = EvalEnv(ticker, pre_processor, training_env, model)
    eval_env.run_pre_processor()
    eval_env.eval_loop()

    init_mlflow(mlflow_dir, "Tests")
    with mlflow.start_run():
        log_file(eval_env.tracker.trades.tracked, fn="trades.csv")
        log_file(eval_env.tracker.env_state.tracked, fn="env_state.csv")
        log_file(eval_env.tracker.tracked, fn="tracked.csv")


main()
