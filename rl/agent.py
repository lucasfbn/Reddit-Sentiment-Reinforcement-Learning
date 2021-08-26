import mlflow
import ray
from tensorforce import Runner, Agent, Environment

import paths
from rl.env import EnvCNN
from rl.pre_env import PreEnv
from utils.mlflow_api import load_file, log_file, MlflowAPI
from utils.util_funcs import log


@ray.remote
def eval_single(agent_path, env, ticker):
    rla = RLAgent(environment=EnvCNN, ticker=None)
    rla.load_agent(agent_path)

    agent = rla.agent

    for sequence in ticker.sequences:
        state = env.shape_state(sequence).df

        action = agent.act(state, independent=True)

        arr = agent.tracked_tensors()["agent/policy/action_distribution/probabilities"]
        actions_proba = {"hold": arr[0], "buy": arr[1], "sell": arr[2]}

        sequence.add_eval(action, actions_proba)

    rla.agent.close()

    return ticker


class RLAgent:

    def __init__(self, environment, ticker):
        self.ticker = ticker

        self.environment = environment
        self.agent = None

        self._agent_saved = False
        self._agent_path = None

    def load_agent(self, artifact_path):
        self.agent = Agent.load(directory=str(artifact_path / "model"), format='numpy', tracking="all")
        self._agent_path = artifact_path
        self._agent_saved = True

    def save_agent(self):
        path = str(MlflowAPI().get_artifact_path() / "model")
        self.agent.save(directory=path, format='numpy')
        self._agent_saved = True

    def eval_agent(self):
        log.info("Evaluating...")
        ray.init(ignore_reinit_error=True)

        env = self.environment
        agent_path = MlflowAPI().get_artifact_path() if self._agent_path is None else self._agent_path

        futures = [eval_single.remote(agent_path=agent_path, env=env, ticker=t) for t in self.ticker]
        evaluated_ticker = ray.get(futures)

        log_file(evaluated_ticker, f"eval.pkl")

        if not self._agent_saved:
            self.save_agent()

        return evaluated_ticker

    def train(self, n_full_episodes):
        pre_env = PreEnv(self.ticker)
        pre_env.exclude_non_tradeable_sequences()
        ticker = pre_env.get_updated_ticker()

        environment = Environment.create(environment=self.environment, ticker=ticker)

        if self.agent is None:
            self.agent = Agent.create(
                agent='ppo', environment=environment, batch_size=32, tracking="all",
                # exploration=0.02
            )

        def log_callback(runner_, _):
            env = runner_.environments[0]
            env.log()

        runner = Runner(agent=self.agent, environment=environment)
        runner.run(num_episodes=int(n_full_episodes * len(ticker)), callback=log_callback,
                   callback_episode_frequency=len(ticker))
        runner.close()

        self.save_agent()

        environment.close()

    def close(self):
        self.agent.close()


if __name__ == '__main__':
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Tests")

    with mlflow.start_run():
        data = load_file(run_id="f4bdae299f694599ba91c7dd1f77c9b5", fn="ticker.pkl", experiment="Datasets")
        rla = RLAgent(environment=EnvCNN, ticker=data)
        # rla.load_agent(MlflowAPI(run_id="230bb130c5314840b557e80d530d692c",
        #                          experiment="Exp: Retrain agent").get_artifact_path())
        rla.train(n_full_episodes=10)
        # rla.eval_agent()
        rla.close()
        mlflow.log_param("parallel", True)
