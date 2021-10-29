import mlflow

import paths
from rl.envs.env import EnvCNN
from rl.envs.simple_trading import SimpleTradingEnv
from rl.wrapper.agent import AgentActObserve
from rl.wrapper.environment import EnvironmentWrapper
from utils.logger import setup_logger
from utils.mlflow_api import load_file


class CustomAgent(AgentActObserve):

    def episode_end_callback(self, episode):
        self.log_callback(self.env.tf_env)
        pred = self.predict(get_probabilities=False)
        self.report_callback(episode, pred)
        del pred


if __name__ == '__main__':
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Tests")

    with mlflow.start_run():
        setup_logger("INFO")
        data = load_file(run_id="5f83fb769d0f4440ab0d13d14fc27e5e", fn="ticker.pkl", experiment="Datasets")

        SimpleTradingEnv.ENABLE_TRANSACTION_COSTS = True
        SimpleTradingEnv.ENABLE_NEG_BUY_REWARD = True
        SimpleTradingEnv.ENABLE_POS_SELL_REWARD = True

        mlflow.log_params(dict(ENABLE_TRANSACTION_COSTS=SimpleTradingEnv.ENABLE_TRANSACTION_COSTS,
                               ENABLE_NEG_BUY_REWARD=SimpleTradingEnv.ENABLE_NEG_BUY_REWARD,
                               ENABLE_POS_SELL_REWARD=SimpleTradingEnv.ENABLE_POS_SELL_REWARD,
                               TRANSACTION_FEE_BID=SimpleTradingEnv.TRANSACTION_FEE_BID,
                               TRANSACTION_FEE_ASK=SimpleTradingEnv.TRANSACTION_FEE_ASK))

        env = EnvironmentWrapper(EnvCNN, data)
        env.create(max_episode_timesteps=max(len(tck) for tck in env.data))

        agent = CustomAgent(env)
        agent.create()
        # agent.load(MlflowAPI(run_id="c3aaa7c52b3f41afb256c4c3ad4376f4").get_artifact_path())
        agent.train(episodes=25, episode_progress_indicator=env.len_data)
        agent.save()

        # pred = agent.predict()
        # log_file(pred, "pred.pkl")
