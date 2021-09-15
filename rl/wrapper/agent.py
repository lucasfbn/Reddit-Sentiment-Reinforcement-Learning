from tensorforce import Runner, Agent, Environment
from utils.mlflow_api import load_file, log_file, MlflowAPI
from tqdm import tqdm


class AgentWrapper:

    def __init__(self, env):
        self.agent = None
        self.env = env

    def create(self, **kwargs):
        self.agent = Agent.create(
            agent='ppo', environment=self.env.tf_env, batch_size=32, tracking="all",
            # exploration=0.02,
            **kwargs
        )

    def load(self, artifact_path):
        self.agent = Agent.load(directory=str(artifact_path / "model"), format='numpy', tracking="all")

    def save(self):
        path = str(MlflowAPI().get_artifact_path() / "model")
        self.agent.save(directory=path, format='numpy')

    def close(self):
        self.agent.close()

    def train(self, episodes, episode_progress_indicator, episode_interval=None):
        raise NotImplementedError

    def episode_start_callback(self):
        pass

    def episode_end_callback(self):
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


class AgentRunner(AgentWrapper):

    def train(self, episodes, episode_progress_indicator, episode_interval=None):
        runner = Runner(agent=self.agent, environment=self.env.tf_env)
        runner.run(num_episodes=episodes, callback=self.in_episode_callback)
        runner.close()


class AgentActObserve(AgentWrapper):

    def train(self, episodes, episode_progress_indicator, episode_interval=None):

        env = self.env.tf_env

        for ep in range(episodes):

            self.episode_start_callback()

            for _ in tqdm(range(episode_progress_indicator), desc=f"Episode {ep}"):

                states = env.reset()
                terminal = False

                while not terminal:
                    actions = self.agent.act(states=states)
                    states, terminal, reward = env.execute(actions=actions)
                    self.agent.observe(terminal=terminal, reward=reward)

            self.episode_end_callback()

            if episode_interval is not None and ep % episode_interval == 0:
                self.episode_end_interval_callback()


class AgentActExperienceUpdate(AgentWrapper):

    def train(self, episodes, episode_progress_indicator, episode_interval=None):

        env = self.env.tf_env

        for ep in range(episodes):

            self.episode_start_callback()

            for _ in tqdm(range(episode_progress_indicator), desc=f"Episode {ep}"):
                self.in_episode_start_callback()

                internals = self.agent.initial_internals()
                episode_states = list()
                episode_internals = list()
                episode_actions = list()
                episode_terminal = list()
                episode_reward = list()

                self.in_episode_callback()

                self.agent.experience(
                    states=episode_states, internals=episode_internals, actions=episode_actions,
                    terminal=episode_terminal, reward=episode_reward
                )

                self.agent.update()

                self.in_episode_end_callback()

            self.episode_end_callback()

            if episode_interval is not None and ep % episode_interval == 0:
                self.episode_end_interval_callback()
