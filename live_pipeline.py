import pickle as pkl
from datetime import datetime

import mlflow

import paths
import preprocessing.config as preprocessing_config
import preprocessing.new_dataset as preprocessing_new_dataset
import sentiment_analysis.config as sentiment_analysis_config
from evaluate.eval_portfolio import EvalLive
from learning_tensorforce.agent import RLAgent
from learning_tensorforce.env import EnvCNN
from preprocessing.dataset_loader import DatasetLoader
from sentiment_analysis.new_dataset import Dataset
from mlflow_api import log_file
from utils import log

log.setLevel("INFO")

mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("Live")


class LivePipeline:

    def __init__(self, agent_path, last_state_path=None):
        self.last_state_path = last_state_path
        self.agent_path = agent_path
        self.data = None

        self.rl_agent = None
        self.optimal_thresholds = None
        self.evaluation = None

    def new_sentiment_dataset(self):
        sentiment_analysis_config.general.start = datetime(year=2021, month=2, day=18)
        now = datetime.now()
        sentiment_analysis_config.general.end = datetime(hour=18, minute=0, day=now.day, month=now.month, year=now.year)
        ds = Dataset(sentiment_analysis_config)
        ds.create()

    def new_dataset(self):
        preprocessing_config.general.from_run_id = mlflow.active_run().info.run_id
        preprocessing_config.merge_preprocessing.live = True
        preprocessing_new_dataset.main(preprocessing_config)

    def load_data(self):
        self.data = DatasetLoader([mlflow.active_run().info.run_id], "cnn").merge()

    def retrain_model(self):
        rla = RLAgent(environment=EnvCNN, train_data=self.data)
        rla.train(n_full_episodes=12)

        self.rl_agent = rla

    def evaluate(self):
        self.optimal_thresholds = self.rl_agent.eval_agent()[1]["thresholds"]
        self.rl_agent.close()

    def get_most_recent_data(self):
        for data in self.data:
            data["data"] = data["data"].tail(1)

    def trade(self):
        if self.evaluation is None:
            self.evaluation = EvalLive(data=self.data, live=True, quantiles_thresholds=self.optimal_thresholds,
                                       fixed_thresholds=True)
        self.evaluation.act()

    def load_state(self):
        with open(self.last_state_path, "rb") as f:
            self.evaluation = pkl.load(f)

    def save_state(self):
        log_file(self.evaluation, "state.pkl")

    def run(self):
        self.new_sentiment_dataset()
        self.new_dataset()
        self.load_data()
        self.retrain_model()
        self.evaluate()
        self.get_most_recent_data()

        if self.last_state_path is not None:
            self.load_state()
        self.trade()
        self.save_state()


if __name__ == "__main__":
    with mlflow.start_run():
        live = LivePipeline("")
        live.run()

# TODO:
# - Test
# - Store state somewhere
# - Daily (full) retrain model
