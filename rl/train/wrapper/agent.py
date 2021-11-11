import mlflow
from mlflow_utils import MlflowUtils
from tensorforce import Runner, Agent
from tqdm import tqdm

from rl.eval.evaluate import Evaluate
from rl.train.wrapper.predict import predict_wrapper


class AgentWrapper:

    def __init__(self, env):
        self.agent = None
        self.env = env

    def create(self, **kwargs):
        self.agent = Agent.create(
            agent='ppo', environment=self.env.tf_env, batch_size=32, tracking="all", memory="minimum",
            # exploration=0.02,
            **kwargs
        )

    def load(self, artifact_path):
        self.agent = Agent.load(directory=str(artifact_path / "model"), format='numpy', tracking="all")

    def save(self):
        path = str(MlflowUtils().get_artifact_path() / "model")
        self.agent.save(directory=path, format='numpy')
        return path

    def close(self):
        self.agent.close()

    def train(self, episodes, episode_progress_indicator, episode_interval=None):
        raise NotImplementedError

    def predict(self, get_probabilities):
        agent_path = self.save()
        pred = predict_wrapper(agent_path=agent_path, env=self.env.env, x=self.env.data, get_probas=get_probabilities)
        return pred

    def episode_start_callback(self, episode):
        pass

    def episode_end_callback(self, episode):
        pass

    def episode_end_interval_callback(self):
        pass

    def in_episode_start_callback(self):
        pass

    def in_episode_callback(self):
        pass

    def in_episode_end_callback(self):
        pass

    @staticmethod
    def log_callback(env):
        env.log()

    def eval_callback(self, pred):
        with mlflow.start_run(nested=True):
            ep = Evaluate(ticker=pred)
            ep.set_thresholds({'hold': 0, 'buy': 0, 'sell': 0})
            ep.initialize()
            ep.act()
            ep.force_sell()
            ep.log_params()
            ep.log_metrics()
            ep.log_statistics()


class AgentRunner(AgentWrapper):

    def runner_callback(self, runner_, _):
        self.log_callback(runner_.environments[0])

    def train(self, episodes, episode_progress_indicator, episode_interval=None):
        episodes *= episode_progress_indicator

        def _runner_callback(runner_, _):
            self.runner_callback(runner_, _)
            return True

        runner = Runner(agent=self.agent, environment=self.env.tf_env)
        runner.run(num_episodes=episodes, callback=_runner_callback,
                   callback_episode_frequency=episode_progress_indicator)

        runner.close()


class AgentActObserve(AgentWrapper):

    def train(self, episodes, episode_progress_indicator, episode_interval=None):

        env = self.env.tf_env

        for ep in range(episodes):

            self.episode_start_callback(ep)

            for _ in tqdm(range(episode_progress_indicator), desc=f"Episode {ep}"):

                states = env.reset()
                terminal = False

                while not terminal:
                    actions = self.agent.act(states=states)
                    states, terminal, reward = env.execute(actions=actions)
                    self.agent.observe(terminal=terminal, reward=reward)

            self.episode_end_callback(ep)

            if episode_interval is not None and ep % episode_interval == 0:
                self.episode_end_interval_callback()


class AgentActExperienceUpdate(AgentWrapper):

    def train(self, episodes, episode_progress_indicator, episode_interval=None):

        env = self.env.tf_env

        for ep in range(episodes):

            self.episode_start_callback(ep)

            internals = self.agent.initial_internals()
            episode_states = list()
            episode_internals = list()
            episode_actions = list()
            episode_terminal = list()
            episode_reward = list()

            for _ in tqdm(range(episode_progress_indicator), desc=f"Episode {ep}"):
                self.in_episode_start_callback()

                states = env.reset()
                terminal = False

                while not terminal:
                    episode_states.append(states)
                    episode_internals.append(internals)
                    actions, internals = self.agent.act(states=states, internals=internals, independent=True)
                    episode_actions.append(actions)
                    states, terminal, reward = env.execute(actions=actions)
                    episode_terminal.append(terminal)
                    episode_reward.append(reward)

                self.in_episode_callback()

            self.agent.experience(
                states=episode_states, internals=episode_internals, actions=episode_actions,
                terminal=episode_terminal, reward=episode_reward
            )

            self.agent.update()

            self.in_episode_end_callback()

            self.episode_end_callback(ep)

            if episode_interval is not None and ep % episode_interval == 0:
                self.episode_end_interval_callback()
