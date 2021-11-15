import mlflow
from mlflow_utils import load_file, init_mlflow, setup_logger

import utils.paths
from rl.eval.envs.env import Evaluate

init_mlflow(utils.paths.mlflow_dir, "Tests")

with mlflow.start_run():
    setup_logger("DEBUG")
    ticker = load_file(run_id="15c90f47b0ce4066a7c8daee597e98a3", experiment="Tests", fn="eval.pkl")

    for t in ticker:
        for s in t.sequences:
            s.actions_proba = {"hold": 1, "buy": 1, "sell": 1}

    combination = {'max_trades_per_day': 3, 'max_price_per_stock': 20, 'max_investment_per_trade': 0.07}

    ep = Evaluate(ticker=ticker, **combination, initial_balance=1000)
    ep.set_thresholds({'hold': 0, 'buy': 0, 'sell': 0})
    ep.initialize()
    # ep.set_quantile_thresholds({'hold': None, 'buy': 0.95, 'sell': None})
    ep.act()
    ep.force_sell()
    ep.log_results()
    ep.log_statistics()
    # print(ep.get_result())
