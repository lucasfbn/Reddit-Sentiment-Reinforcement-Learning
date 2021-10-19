import mlflow

import paths
from rl.envs.env import EnvCNN
from rl.wrapper.agent import AgentActExperienceUpdate, AgentActObserve
from rl.wrapper.environment import EnvironmentWrapper
from utils.mlflow_api import load_file, MlflowAPI, log_file
from utils.logger import setup_logger


class CustomAgent(AgentActObserve):

    def episode_end_callback(self):
        self.log_callback(self.env.tf_env)
        self.eval_callback()


if __name__ == '__main__':
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Tests")

    with mlflow.start_run():
        setup_logger("INFO")
        data = load_file(run_id="582dd57030d74e7d8cf4f8fd2b1fb189", fn="ticker.pkl", experiment="Datasets")

        env = EnvironmentWrapper(EnvCNN, data)
        env.create(max_episode_timesteps=max(len(tck) for tck in env.data))

        agent = CustomAgent(env)
        # agent.create()
        agent.load(MlflowAPI(run_id="c3aaa7c52b3f41afb256c4c3ad4376f4").get_artifact_path())

        # agent.train(episodes=6, episode_progress_indicator=env.len_data)
        # agent.save()
        pred = agent.predict()
        log_file(pred, "pred.pkl")
