from tensorforce import Runner, Agent, Environment


class EnvironmentWrapper:

    def __init__(self, env, data):
        self.env = env
        self.data = data
        self.len_data = len(data)

        self.tf_env = None

    def create(self, **kwargs):
        self.tf_env = Environment.create(environment=self.env, ticker=self.data, **kwargs)
