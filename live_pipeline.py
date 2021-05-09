import pickle as pkl
from datetime import datetime

import mlflow

import paths
import preprocessing.config as preprocessing_config
import preprocessing.new_dataset as preprocessing_main
import sentiment_analysis.config as sentiment_analysis_config
from evaluate.eval_portfolio import EvalLive
from learning_tensorforce.agent import RLAgent
from learning_tensorforce.env import EnvCNN
from preprocessing.dataset_loader import DatasetLoader
from sentiment_analysis.new_dataset import Dataset

mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("Live")
mlflow.start_run()


class LivePipeline:

    def __init__(self):
        self.data = None

        self.rl_agent = None
        self.optimal_thresholds = None
        self.evaluation = None

    def new_sentiment_dataset(self):
        sentiment_analysis_config.general.start = datetime(year=2021, month=2, day=18)
        now = datetime.now()
        sentiment_analysis_config.general.end = datetime(hour=15, minute=0, day=now.day, month=now.month, year=now.year)
        ds = Dataset(sentiment_analysis_config)
        ds.create()

    def new_dataset(self):
        preprocessing_config.general.from_run_id = mlflow.active_run().info.run_id
        preprocessing_config.merge_preprocessing.live = True
        preprocessing_main.main(preprocessing_config)

    def load_data(self):
        self.data = DatasetLoader([mlflow.active_run().info.run_id], "cnn").merge()

    def retrain_model(self, agent_path):
        rla = RLAgent(environment=EnvCNN, train_data=self.data)
        rla.load_agent(agent_path)
        rla.train(n_full_episodes=2)

        self.rl_agent = rla

    def evaluate(self):
        self.optimal_thresholds = self.rl_agent.eval_agent()[1]
        self.rl_agent.close()

    def get_most_recent_data(self):
        for data in self.data:
            data["data"] = data["data"].tail(1)

    def trade(self):
        if self.evaluation is None:
            self.evaluation = EvalLive(eval_data=self.data, live=True, quantiles_thresholds=self.optimal_thresholds)
        self.evaluation.act()

    def load_state(self):
        with open("state.pkl", "rb") as f:
            self.evaluation = pkl.load(f)

    def save_state(self):
        with open("state.pkl", "wb") as f:
            pkl.dump(self.evaluation, f)

# TODO:
# - Test
# - Store state somewhere
# - Daily (full) retrain model
