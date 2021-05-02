import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorforce import Runner, Agent, Environment
from learning_tensorforce.env import EnvNN, EnvCNN
from evaluate.eval_portfolio import EvaluatePortfolio

from tqdm import tqdm

from utils import mlflow_log_file
import paths
import pickle as pkl
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
        self.agent = Agent.load(directory=artifact_path + "/model", format='numpy')
        self.artifact_path = self.artifact_path

    def save_agent(self):
        if self.artifact_path is not None:
            path = self.artifact_path + "/model"
            self.agent.save(directory=path, format='numpy')
            self._agent_saved = True

    def _eval(self, data, suffix):

        env = self.environment()

        for grp in tqdm(data):

            actions = []
            actions_outputs = []

            for state in grp["data"]:
                state = env._shape_state(state)
                action = self.agent.act(state, independent=True)
                actions.append(action)
                actions_outputs.append(1)

            grp["metadata"]["actions"] = actions
            grp["metadata"]["actions_outputs"] = actions_outputs

        if self.artifact_path is not None:
            mlflow_log_file(data, f"eval_{suffix}.pkl")

        ep = EvaluatePortfolio(data, max_buy_output=2)
        ep.act()
        ep.force_sell()

        mlflow.log_metrics({f"Profit_{suffix}": ep.profit, f"Balance_{suffix}": ep.balance,
                            f"Index_perf_{suffix}": data[0]["index_comparison"]["perf"]})

        if not self._agent_saved:
            self.save_agent()

    def eval_agent(self):
        # self._eval(self.train_data, "train")

        if self.test_data is not None:

            for i, data in enumerate(self.test_data):
                self._eval(data, f"test_{i}")

    def train(self, n_full_episodes):
        self.environment.data = self.train_data
        environment = Environment.create(environment=self.environment)

        self.agent = Agent.create(
            agent='ppo', environment=environment,
            memory=3000, batch_size=32,
            exploration=0.01
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

    rla = RLAgent(environment=EnvCNN, train_data=training_data, test_data=test_data)
    rla.train(n_full_episodes=20)
    rla.eval_agent()
    rla.close()
    # rla.load_agent("C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/5/ed688c07f09c4daebb854e7badccc0a7/artifacts/")
    # rla.eval_agent()

    mlflow.end_run()
