import mlflow
import pandas as pd
import tensorflow as tf
from tensorforce import Runner, Agent, Environment
from tqdm import tqdm

import paths
from rl.env import EnvCNN
from utils.mlflow_api import load_file
from utils.mlflow_api import log_file
from utils.util_funcs import log

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class RLAgent:

    def __init__(self, environment, ticker):
        self.ticker = ticker

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

    @staticmethod
    def _merge_actions(actions, prices, actions_outputs):
        df = pd.DataFrame(dict(actions=actions, prices=prices)).reset_index(drop=True)
        actions_outputs = pd.concat(actions_outputs, axis="rows").reset_index(drop=True)
        df = pd.concat([df, actions_outputs], axis="columns")
        return df

    def _eval(self):
        log.info("Evaluating...")

        env = self.environment()

        for ticker in tqdm(self.ticker, "Processing ticker "):

            prices = []
            actions = []
            actions_outputs = []

            for sequence in env.get_sequences(ticker):
                state = env._shape_state(sequence).df

                action = self.agent.act(state, independent=True)
                action_proba = pd.DataFrame(
                    [self.agent.tracked_tensors()["agent/policy/action_distribution/probabilities"]],
                    columns=["hold_probability", "buy_probability", "sell_probability"])

                prices.append(sequence.price)
                actions.append(action)
                actions_outputs.append(action_proba)

            ticker.add_eval(self._merge_actions(actions, prices, actions_outputs))

    def eval_agent(self):
        self._eval()

        if self.artifact_path is not None:
            log_file(self.ticker, f"eval.pkl")

        if not self._agent_saved:
            self.save_agent()

    def train(self, n_full_episodes):
        self.environment.ticker = self.ticker
        environment = Environment.create(environment=self.environment)

        self.agent = Agent.create(
            agent='ppo', environment=environment,
            memory=3000, batch_size=32,
            exploration=0.01,
            tracking="all"
        )

        runner = Runner(agent=self.agent, environment=environment)
        runner.run(num_episodes=int(n_full_episodes * len(self.ticker)))
        runner.close()

        self.save_agent()

        environment.close()

    def close(self):
        self.agent.close()


if __name__ == '__main__':
    training_data = load_file(run_id="cdd0ea6c04d64b009dc1ebdeabcba818", fn="ticker.pkl", experiment="Tests")

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Learning")

    with mlflow.start_run():
        rla = RLAgent(environment=EnvCNN, ticker=training_data)
        # rla.load_agent(
        #     "C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/5/56f707cead8140e782f712752ff21fad/artifacts")
        rla.train(n_full_episodes=1)
        rla.eval_agent()
        rla.close()
