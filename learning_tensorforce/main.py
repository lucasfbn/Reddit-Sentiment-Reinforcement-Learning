import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorforce import Runner, Agent, Environment
from learning_tensorforce.env import EnvNN
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

        ep = EvaluatePortfolio(data)
        ep.act()
        ep.force_sell()

        mlflow.log_metrics({f"Profit_{suffix}": ep.profit, f"Balance_{suffix}": ep.balance,
                            f"Index_perf_{suffix}": data["index_comparison"]["perf"]})

        if not self._agent_saved:
            self.save_agent()

    def eval_agent(self):
        self._eval(self.train_data, "train")

        if self.test_data is not None:

            for i, test_data in enumerate(self.test_data):
                self._eval(self.test_data, f"test_{i}")

    def train(self):
        EnvNN.data = self.train_data
        environment = Environment.create(environment=self.environment)

        self.agent = Agent.create(
            agent='ppo', environment=environment,
            memory=2000, batch_size=32, exploration=0.01
        )

        runner = Runner(agent=self.agent, environment=environment)
        runner.run(num_episodes=2000)
        runner.close()

        self.save_agent()
        self.eval_agent()

        self.agent.close()
        environment.close()


if __name__ == '__main__':
    training_ids = ["58d6faa3746b46b7839f62fcb03239ea", "82e8310bd4134c45a071ce6d5175b297",
                    "7caf13efeaf14d879bbcd693143e2b8a"]
    test_ids = ["b1af9027e200433f880122645cab22eb", "fd8fde5f6325439d91080feca3731aa9",
                "0149e640c1c94b90bf7816754343a521"]

    training_data = DatasetLoader(training_ids).load().merge()
    test_data = DatasetLoader(test_ids).load()

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Learning")
    mlflow.start_run()

    rla = RLAgent(environment=EnvNN, train_data=training_data, test_data=test_data)
    # rla.train()
    # rla.eval_agent()

    mlflow.end_run()
