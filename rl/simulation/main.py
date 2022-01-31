from rl.simulation.envs.sim import Simulation
from rl.simulation.envs.pre_process.pre_process import PreProcessor
import wandb
from dataset_handler.stock_dataset import StockDatasetWandb
from utils.wandb_utils import log_file


def main():
    with wandb.init(project="Trendstuff", group="Throwaway") as run:
        dataset = StockDatasetWandb()
        dataset.wandb_load_meta_file("as6t2wi1", run)

        dataset = dataset
        eval_env = Simulation(dataset)
        eval_env.prepare_data()
        eval_env.eval_loop()
        #
        log_file(eval_env.detail_tracker.trades.tracked, fn="detailed_trades.csv", run=run)
        log_file(eval_env.detail_tracker.env_state.tracked, fn="detailed_env_state.csv", run=run)
        log_file(eval_env.detail_tracker.tracked, fn="detailed_tracked.csv", run=run)
        log_file(eval_env.overall_tracker.tracked, fn="overall_tracked.csv", run=run)


main()
