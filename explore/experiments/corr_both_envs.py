import mlflow
from tensorforce import Agent, Environment
from tqdm import tqdm

import paths
from eval.evaluate import Evaluate
from rl.agent import RLAgent
from rl.envs.env import EnvCNN
from utils.mlflow_api import MlflowAPI
from utils.mlflow_api import load_file
from utils.util_funcs import log

"""
Examines whether the results of training and the evaluation environment are correlating
"""

log.setLevel("INFO")

mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("Exp: Corr Both Envs")

n_episodes = 10


class RLAgentEvalEveryEpisode(RLAgent):
    def train(self, n_full_episodes):
        self.environment.data = self.ticker
        environment = Environment.create(environment=self.environment)

        if self.agent is None:
            self.agent = Agent.create(
                agent='ppo', environment=environment, batch_size=32, tracking="all",
                # exploration=0.02
            )

        # runner = Runner(agent=self.agent, environment=environment)
        # runner.run(num_episodes=int(n_full_episodes * len(self.ticker)))
        # runner.close()

        full_episode_counter = 0

        for i in tqdm(range(int(n_full_episodes * len(self.ticker)))):

            states = environment.reset()
            terminal = False
            while not terminal:
                actions = self.agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                self.agent.observe(terminal=terminal, reward=reward)

            if i != 0 and i % len(self.ticker) == 0:
                with mlflow.start_run(nested=True):
                    full_episode_counter += 1
                    self.artifact_path = MlflowAPI().get_artifact_path()
                    self._agent_saved = False

                    self.save_agent()

                    evaluated = self.eval_agent()

                    ep = Evaluate(ticker=evaluated)
                    ep.activate_training_emulator()
                    ep.initialize()
                    ep.act()
                    ep.force_sell()
                    ep.log_results()
                    # ep.log_statistics()
                    mlflow.log_metric("episode", full_episode_counter)

        self.artifact_path = MlflowAPI().get_artifact_path()
        self.save_agent()

        environment.close()


with mlflow.start_run():
    data = load_file(run_id="f4bdae299f694599ba91c7dd1f77c9b5", fn="ticker.pkl", experiment="Datasets")
    # data = data[:100]

    # Train
    rla = RLAgentEvalEveryEpisode(environment=EnvCNN, ticker=data)
    rla.train(n_full_episodes=n_episodes)

    # Eval
    rla.eval_agent()
    rla.close()
