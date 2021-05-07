import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorforce import Runner, Agent, Environment
from learning_tensorforce.env import EnvCNN
from evaluate.eval_portfolio import EvaluatePortfolio
import pandas as pd
from tqdm import tqdm

from utils import mlflow_log_file, log
import paths
import mlflow

from preprocessing.dataset_loader import DatasetLoader


class RLAgent:

    def __init__(self, environment, train_data, test_data=None):
        self.train_data = train_data
        self.test_data = test_data

        self.artifact_path = None if mlflow.active_run() is None else "C:" + mlflow.get_artifact_uri().split(":")[2]

        self.environment = environment
        self.agent = None

        self._agent_saved = False

    def load_agent(self, artifact_path):
        self.agent = Agent.load(directory=artifact_path + "/model", format='numpy', tracking="all")
        self.artifact_path = self.artifact_path

    def save_agent(self):
        if self.artifact_path is not None:
            path = self.artifact_path + "/model"
            self.agent.save(directory=path, format='numpy')
            self._agent_saved = True

    def _eval(self, data, suffix):
        log.info("Evaluating...")
        env = self.environment()

        for grp in tqdm(data):

            actions = []
            actions_outputs = []

            for state in grp["data"]:
                state = env._shape_state(state)
                a = self.agent.tracked_tensors()
                action = self.agent.act(state, independent=True)
                actions.append(action)

                probas = pd.DataFrame([self.agent.tracked_tensors()["agent/policy/action_distribution/probabilities"]],
                                      columns=["hold_probability", "buy_probability", "sell_probability"])
                actions_outputs.append(probas)

            grp["metadata"]["actions"] = actions
            grp["metadata"] = grp["metadata"].reset_index(drop=True)
            actions_outputs = pd.concat(actions_outputs, axis="rows").reset_index(drop=True)
            grp["metadata"] = pd.concat([grp["metadata"], actions_outputs], axis="columns")

        if self.artifact_path is not None:
            mlflow_log_file(data, f"eval_{suffix}.pkl")

        ep = EvaluatePortfolio(data)
        ep.act()
        ep.force_sell()

        mlflow.log_metrics({f"Profit_{suffix}": ep.profit, f"Balance_{suffix}": ep.balance,
                            f"Index_perf_{suffix}": data[0]["index_comparison"]["perf"]})

        if not self._agent_saved:
            self.save_agent()

    def eval_agent(self):
        for i, data in enumerate(self.train_data):
            self._eval(data, f"train_{i}")

        if self.test_data is not None:
            for i, data in enumerate(self.test_data):
                self._eval(data, f"test_{i}")

    def train(self, n_full_episodes):
        self.environment.data = self.train_data
        environment = Environment.create(environment=self.environment)

        self.agent = Agent.create(
            agent='ppo', environment=environment,
            memory=3000, batch_size=32,
            exploration=0.01,
            tracking="all"
        )

        runner = Runner(agent=self.agent, environment=environment)
        runner.run(num_episodes=n_full_episodes * len(self.train_data))
        runner.close()

        self.save_agent()

        environment.close()

    def close(self):
        self.agent.close()


if __name__ == '__main__':
    training_ids = ["989a7b1534144304b0575bf964a88ad3"]

    training_data = DatasetLoader(training_ids, "cnn").merge()
    test_data = DatasetLoader(training_ids, "cnn").load()

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Learning")
    mlflow.start_run()

    rla = RLAgent(environment=EnvCNN, train_data=training_data)
    rla.load_agent(
        "C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/5/56f707cead8140e782f712752ff21fad/artifacts")
    # rla.train(n_full_episodes=1)
    rla.eval_agent()
    rla.close()
    # rla.eval_agent()

    mlflow.end_run()
