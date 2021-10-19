from datetime import datetime

import mlflow

import paths
from eval.evaluate import EvalLive
from preprocessing.pipeline import pipeline as preprocessing_pipeline
from rl.agent import RLAgent
from rl.envs.env import EnvCNN
from sentiment_analysis.pipeline import pipeline as sentiment_analysis_pipeline
from utils.mlflow_api import load_file
from utils.logger import setup_logger


class LivePipeline:
    live = True

    def __init__(self, last_state_path=None):
        self.last_state_path = last_state_path

        self.sentiment_analysis_result = None
        self.preprocessing_result = None

        self.ticker = None
        self.evaluated = None

        self.rl_agent = None
        self.optimal_thresholds = None
        self.evaluation = None

    def run_sentiment_analysis(self):
        now = datetime.now()
        self.sentiment_analysis_result = sentiment_analysis_pipeline(start=datetime(year=2021, month=3, day=2),
                                                                     end=datetime(hour=0, minute=0, day=13,
                                                                                  month=8, year=2021))

    def run_preprocessing(self):
        self.ticker = preprocessing_pipeline(input_df=load_file("report.csv"),
                                             enable_live_behaviour=self.live)

    def train(self):
        rla = RLAgent(environment=EnvCNN, ticker=load_file("ticker.pkl"))
        rla.train(n_full_episodes=7)
        rla.close()

    def evaluate(self):

        # We are only interested in the last sequence
        for t in self.ticker:
            t.sequences = [t.sequences[len(t.sequences) - 1]]

        rla = RLAgent(environment=EnvCNN, ticker=self.ticker)
        rla.load_agent(get_artifact_path())
        self.evaluated = rla.eval_agent()
        rla.close()

    def trade(self):
        combination = {'max_trades_per_day': 3, 'max_price_per_stock': 20,
                       'max_investment_per_trade': 0.07}
        self.evaluation = EvalLive(ticker=self.evaluated, live=self.live,
                                   **combination)
        if self.last_state_path is not None:
            self.evaluation.load(self.last_state_path)
        self.evaluation.set_thresholds({'hold': 0, 'buy': 0, 'sell': 0})
        self.evaluation.initialize()
        self.evaluation.act()
        self.evaluation.save()

    def run(self):
        self.run_sentiment_analysis()
        self.run_preprocessing()
        # self.train()
        # self.evaluate()
        # self.trade()


if __name__ == "__main__":
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Live")

    with mlflow.start_run():
        log = setup_logger(level="DEBUG")
        live = LivePipeline()
        # live.ticker = load_file(run_id="662f377d540e42f68f2df688c24a060c", fn="ticker.pkl", experiment="Live")
        # live.evaluated = load_file("eval.pkl")
        live.run()
