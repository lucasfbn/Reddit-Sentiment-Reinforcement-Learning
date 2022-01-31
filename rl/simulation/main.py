import wandb

from dataset_handler.stock_dataset import StockDatasetWandb
from rl.simulation.envs.sim import SimulationWandb


def main(meta_file_run_id):
    with wandb.init(project="Trendstuff", group="Simulation") as run:
        dataset = StockDatasetWandb()
        dataset.wandb_load_meta_file(meta_file_run_id, run)

        dataset = dataset
        eval_env = SimulationWandb(dataset)
        eval_env.prepare_data()
        eval_env.eval_loop()

        wandb.config.update(dict(
            meta_file_run_id=meta_file_run_id,
            start_balance=eval_env.trading_env.START_BALANCE,
            investment_per_trade=eval_env.trading_env.INVESTMENT_PER_TRADE,
            max_price_per_stock=eval_env.trading_env.MAX_PRICE_PER_STOCK,
            slippage=eval_env.trading_env.SLIPPAGE,
            order_fee=eval_env.trading_env.ORDER_FEE
        ))


main("as6t2wi1")
