import mlflow
from rl.agent import RLAgent

from rl.eval.envs.env import Evaluate
from rl.train.envs.env import EnvCNN
from utils import paths
from utils.mlflow_api import load_file
from utils.util_funcs import log

"""
Examines whether the reward (e.g. balance in eval) is constant among several runs at 1 episode.
Issue: #81
"""

log.setLevel("INFO")

mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("Exp: Constant Reward 1 Episode, exploration on")

n_episodes = [1] * 5

for ne in n_episodes:
    with mlflow.start_run():
        data = load_file(run_id="f4bdae299f694599ba91c7dd1f77c9b5", fn="ticker.pkl", experiment="Datasets")

        # Train
        rla = RLAgent(environment=EnvCNN, ticker=data)
        rla.train(n_full_episodes=ne)

        # Eval
        rla.eval_agent()
        rla.close()

        # Simulate
        ticker = load_file(fn="eval.pkl")
        combination = {'max_trades_per_day': 3, 'max_price_per_stock': 20, 'max_investment_per_trade': 0.07}

        ep = Evaluate(ticker=ticker, **combination)
        ep.initialize()
        ep.set_quantile_thresholds({'hold': None, 'buy': 0.95, 'sell': None})
        ep.act()
        ep.force_sell()
        ep.log_results()
        ep.log_statistics()

        mlflow.log_param("Trained episodes", ne)
        mlflow.log_param("Exploration on", True)

# os.system("shutdown /s /t 60")
