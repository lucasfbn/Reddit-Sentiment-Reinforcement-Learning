import mlflow
from tensorforce import Runner, Agent, Environment
from tqdm import tqdm

import paths
from rl.env import EnvCNN
from utils.mlflow_api import load_file
from utils.mlflow_api import log_file
from utils.util_funcs import log


class RLAgent:

    def __init__(self, environment, ticker):
        self.ticker = ticker

        self.artifact_path = None if mlflow.active_run() is None else "C:" + mlflow.get_artifact_uri().split(":")[2]

        self.environment = environment
        self.agent = None

        self._agent_saved = False

    def load_agent(self, artifact_path):
        self.agent = Agent.load(directory=str(artifact_path) + "/model", format='numpy', tracking="all")
        self.artifact_path = self.artifact_path

    def save_agent(self):
        if self.artifact_path is not None:
            path = self.artifact_path + "/model"
            self.agent.save(directory=path, format='numpy')
            self._agent_saved = True

    def _eval(self):
        log.info("Evaluating...")

        env = self.environment()

        for ticker in tqdm(self.ticker, "Processing ticker "):

            for sequence in env.get_sequences(ticker):
                state = env._shape_state(sequence).df

                action = self.agent.act(state, independent=True)

                arr = self.agent.tracked_tensors()["agent/policy/action_distribution/probabilities"]
                actions_proba = {"hold": arr[0], "buy": arr[1], "sell": arr[2]}

                sequence.add_eval(action, actions_proba)
                # sequence.cleanup()

    def eval_agent(self):
        self._eval()

        if self.artifact_path is not None:
            log_file(self.ticker, f"eval.pkl")

        if not self._agent_saved:
            self.save_agent()

        return self.ticker

    def train(self, n_full_episodes):
        self.environment.data = self.ticker
        environment = Environment.create(environment=self.environment)

        self.agent = Agent.create(
            agent='ppo', environment=environment,
            memory=55000, batch_size=1, tracking="all",
        )

        runner = Runner(agent=self.agent, environment=environment)
        runner.run(num_episodes=int(n_full_episodes * len(self.ticker)))
        runner.close()

        self.save_agent()

        environment.close()

    def close(self):
        self.agent.close()


if __name__ == '__main__':
    data = load_file(run_id="662f377d540e42f68f2df688c24a060c", fn="ticker.pkl", experiment="Live")

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Tests")

    with mlflow.start_run():
        rla = RLAgent(environment=EnvCNN, ticker=data)
        # rla.load_agent(
        #     "C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/5/56f707cead8140e782f712752ff21fad/artifacts")
        rla.train(n_full_episodes=3)
        rla.eval_agent()
        rla.close()
