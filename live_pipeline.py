import pickle as pkl
from datetime import datetime

import mlflow

import paths
import preprocessing.config as preprocessing_config
import preprocessing.new_dataset as preprocessing_new_dataset
from evaluate.eval_portfolio import EvalLive
from learning_tensorforce.agent import RLAgent
from learning_tensorforce.env import EnvCNN
from preprocessing.dataset_loader import DatasetLoader
from sentiment_analysis.sentiment_analysis_pipeline import flow as sentiment_analysis_flow
from mlflow_api import log_file
from utils import log
from evaluate.cross_validate_evaluation import ParameterTuning, Interval, Choice

log.setLevel("INFO")

mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("Live")


class LivePipeline:

    def __init__(self, last_state_path=None):
        self.last_state_path = last_state_path
        self.data = None

        self.rl_agent = None
        self.optimal_thresholds = None
        self.evaluation = None

    def new_sentiment_dataset(self):
        sentiment_analysis_flow.run(dict(start=datetime(year=2021, month=2, day=18),
                                         end=datetime(year=2021, month=2, day=20)))

    def new_dataset(self):
        preprocessing_config.general.from_run_id = mlflow.active_run().info.run_id
        preprocessing_config.merge_preprocessing.live = True
        preprocessing_new_dataset.main(preprocessing_config)

    def load_data(self):
        self.data = DatasetLoader([mlflow.active_run().info.run_id], "cnn").merge()

    def retrain_model(self):
        rla = RLAgent(environment=EnvCNN, train_data=self.data)
        rla.train(n_full_episodes=13)
        self.data = rla.eval_agent()
        rla.close()

    def tune(self):
        pt = ParameterTuning(self.data,
                             parameter={"buy": Interval(0.7, 1, 0.01)},
                             n_worker=10)
        pt.tune()
        pt.log_top_results(10)

    def get_most_recent_data(self):
        for data in self.data:
            data["metadata"] = data["metadata"].tail(1)

    def trade(self):
        combination = {'max_trades_per_day': 3, 'max_price_per_stock': 20,
                       'max_investment_per_trade': 0.05,
                       'quantiles_thresholds': {'hold': 0, 'buy': 0.9989775836467742, 'sell': 0}}
        self.evaluation = EvalLive(data=self.data, live=True, fixed_thresholds=True,
                                   **combination)
        if self.last_state_path is not None:
            self.evaluation.load(self.last_state_path)

        self.evaluation.initialize()
        self.evaluation.act()

    def save_state(self):
        self.evaluation.save()

    def run(self):
        # self.new_sentiment_dataset()
        # self.new_dataset()
        # self.load_data()
        # self.retrain_model()

        with open("C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/7/"
                  "731bc8beab63455e9186ca304b1ae094/artifacts/eval_train.pkl", "rb") as f:
            self.data = pkl.load(f)
        # self.tune()
        #
        self.get_most_recent_data()
        self.trade()
        self.save_state()


if __name__ == "__main__":
    with mlflow.start_run(run_id="731bc8beab63455e9186ca304b1ae094"):
        live = LivePipeline()
        live.run()

# TODO:
# - Test
# - Store state somewhere
# - Daily (full) retrain model
