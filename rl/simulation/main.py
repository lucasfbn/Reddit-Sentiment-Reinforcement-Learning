import wandb

from dataset_handler.stock_dataset import StockDatasetWandb
from rl.simulation.envs.sim import SimulationWandb


def main():
    with wandb.init(project="Trendstuff", group="Throwaway") as run:
        dataset = StockDatasetWandb()
        dataset.wandb_load_meta_file("as6t2wi1", run)

        dataset = dataset
        eval_env = SimulationWandb(dataset)
        eval_env.prepare_data()
        eval_env.eval_loop()


main()
