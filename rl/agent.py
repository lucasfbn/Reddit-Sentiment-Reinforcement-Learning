import mlflow
import ray
from tensorforce import Runner, Agent, Environment
from tqdm import tqdm

import paths
from eval.evaluate import Evaluate
from rl.env import EnvCNN
from rl.pre_env import PreEnv
from utils.mlflow_api import load_file, log_file, MlflowAPI
from utils.util_funcs import log

log.setLevel("DEBUG")


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
        self.env_raw = environment

        self.env_wrapped = None
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

        env = self.env_raw
        agent_path = MlflowAPI().get_artifact_path() if self._agent_path is None else self._agent_path

        futures = [eval_single.remote(agent_path=agent_path, env=env, ticker=t) for t in self.ticker]
        evaluated_ticker = ray.get(futures)

        log_file(evaluated_ticker, f"eval.pkl")

        if not self._agent_saved:
            self.save_agent()

        return evaluated_ticker

    def run_pre_env(self):
        pre_env = PreEnv(self.ticker)
        pre_env.exclude_non_tradeable_sequences()
        self.ticker = pre_env.get_updated_ticker()

    def create_env(self):
        self.env_wrapped = Environment.create(environment=self.env_raw, ticker=self.ticker)
        return self.env_wrapped

    def create_agent(self):
        if self.agent is None:
            self.agent = Agent.create(
                agent='ppo', environment=self.env_wrapped, batch_size=32, tracking="all",
                # exploration=0.02
            )
        return self.agent

    def initialize(self):
        self.create_env()
        self.create_agent()

    def train_via_runner(self, n_full_episodes):

        def log_callback(runner_, _):
            env = runner_.environments[0]
            env.log()

        runner = Runner(agent=self.agent, environment=self.env_wrapped)
        runner.run(num_episodes=int(n_full_episodes * len(self.ticker)), callback=log_callback,
                   callback_episode_frequency=len(self.ticker))
        runner.close()

    def _evaluate_callback(self):
        with mlflow.start_run(nested=True):
            self._agent_saved = False
            self.save_agent()

            evaluated = self.eval_agent()

            combination = {'max_trades_per_day': 3, 'max_price_per_stock': 20, 'max_investment_per_trade': 70,
                           "initial_balance": 1000}

            ep = Evaluate(ticker=evaluated, **combination)
            ep.set_thresholds({'hold': 0, 'buy': 0, 'sell': 0})
            ep.initialize()
            ep.act()
            ep.force_sell()
            ep.log_params()

        ep.log_metrics(step=self.env_wrapped.episode_count)

    def train_custom_loop(self, n_full_episodes):

        def log_callback(env):
            env.log()

        for i in tqdm(range(int(n_full_episodes * len(self.ticker)))):

            states = self.env_wrapped.reset()
            terminal = False
            while not terminal:
                actions = self.agent.act(states=states)
                states, terminal, reward = self.env_wrapped.execute(actions=actions)
                self.agent.observe(terminal=terminal, reward=reward)

            # On the end of every "full" episode (e.g. one iteration through the ticker)
            if i != 0 and i % len(self.ticker) == 0:
                log_callback(self.env_wrapped)
                self._evaluate_callback()

    def close(self):
        self.agent.close()
        self.env_wrapped.close()


if __name__ == '__main__':
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Tests")

    with mlflow.start_run():
        data = load_file(run_id="f4bdae299f694599ba91c7dd1f77c9b5", fn="ticker.pkl", experiment="Datasets")
        rla = RLAgent(environment=EnvCNN, ticker=data)
        rla.run_pre_env()
        rla.initialize()
        rla.train_custom_loop(n_full_episodes=3)
        # rla.load_agent(MlflowAPI(run_id="230bb130c5314840b557e80d530d692c",
        #                          experiment="Exp: Retrain agent").get_artifact_path())
        rla.close()
