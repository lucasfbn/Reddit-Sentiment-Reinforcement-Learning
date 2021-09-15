import mlflow

import paths
from rl.envs.env import EnvCNN
from rl.wrapper.agent import AgentActObserve, log_callback
from rl.wrapper.environment import EnvironmentWrapper
from utils.mlflow_api import load_file


class AgentActObserveCustom(AgentActObserve):

    def episode_end_callback(self):
        log_callback(self.env.tf_env)


if __name__ == '__main__':
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Tests")

    with mlflow.start_run():
        data = load_file(run_id="f4bdae299f694599ba91c7dd1f77c9b5", fn="ticker.pkl", experiment="Datasets")

        env = EnvironmentWrapper(EnvCNN, data)
        env.create(max_episode_timesteps=max(len(tck) for tck in env.data))

        agent = AgentActObserveCustom(env)
        agent.create()

        agent.train(episodes=2, episode_progress_indicator=env.len_data)
