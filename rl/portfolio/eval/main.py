import mlflow
from mlflow_utils import init_mlflow, load_file, log_file

from rl.portfolio.eval.envs.env import EvalEnv
from rl.portfolio.eval.envs.pre_process.pre_process import PreProcessor
from utils.paths import mlflow_dir


def main():
    init_mlflow(mlflow_dir, "Tests")
    ticker = load_file(
        run_id="f384f58217114433875eda44495272ad",
        fn="evl_ticker.pkl",
        experiment="Eval_Stocks",
    )
    pre_processor = PreProcessor()

    eval_env = EvalEnv(ticker, pre_processor)
    eval_env.run_pre_processor()
    eval_env.eval_loop()

    init_mlflow(mlflow_dir, "Tests")
    with mlflow.start_run():
        log_file(eval_env.detail_tracker.trades.tracked, fn="detailed_trades.csv")
        log_file(eval_env.detail_tracker.env_state.tracked, fn="detailed_env_state.csv")
        log_file(eval_env.detail_tracker.tracked, fn="detailed_tracked.csv")
        log_file(eval_env.overall_tracker.tracked, fn="overall_tracked.csv")


main()
