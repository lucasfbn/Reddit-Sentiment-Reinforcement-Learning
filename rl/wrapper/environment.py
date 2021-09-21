from tensorforce import Runner, Agent, Environment


class EnvironmentWrapper:

    def __init__(self, env, data):
        self.env = env
        self.data = data
        self.len_data = len(data)

        self.tf_env = None

    def create(self, max_episode_timesteps, **kwargs):
        self.tf_env = Environment.create(environment=self.env, ticker=self.data,
                                         max_episode_timesteps=max_episode_timesteps, **kwargs)

    def close(self):
        self.tf_env.close()
