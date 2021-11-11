import mlflow
from mlflow_utils import load_file, init_mlflow, setup_logger

from rl.train.envs.env import EnvCNN
from rl.train.envs.simple_trading import SimpleTradingEnvTraining
from rl.train.wrapper.agent import AgentActObserve
from rl.train.wrapper.environment import EnvironmentWrapper


class CustomAgent(AgentActObserve):

    def episode_end_callback(self, episode):
        self.log_callback(self.env.tf_env)

        if episode % 10 == 0:
            pred = self.predict(get_probabilities=False)
            self.report_callback(episode, pred)


if __name__ == '__main__':
    init_mlflow("Optimize_Env")

    with mlflow.start_run():
        setup_logger("INFO")
        data = load_file(run_id="5896df8aa22c41a3ade34d747bc9ed9a", fn="ticker.pkl", experiment="Datasets")

        SimpleTradingEnvTraining.ENABLE_TRANSACTION_COSTS = True
        SimpleTradingEnvTraining.ENABLE_NEG_BUY_REWARD = True
        SimpleTradingEnvTraining.ENABLE_POS_SELL_REWARD = True
        SimpleTradingEnvTraining.PARTIAL_HOLD_REWARD = 0.25

        mlflow.log_params(dict(ENABLE_TRANSACTION_COSTS=SimpleTradingEnvTraining.ENABLE_TRANSACTION_COSTS,
                               ENABLE_NEG_BUY_REWARD=SimpleTradingEnvTraining.ENABLE_NEG_BUY_REWARD,
                               ENABLE_POS_SELL_REWARD=SimpleTradingEnvTraining.ENABLE_POS_SELL_REWARD,
                               TRANSACTION_FEE_BID=SimpleTradingEnvTraining.TRANSACTION_FEE_BID,
                               TRANSACTION_FEE_ASK=SimpleTradingEnvTraining.TRANSACTION_FEE_ASK,
                               HOLD_REWARD_MULTIPLIER=SimpleTradingEnvTraining.HOLD_REWARD_MULTIPLIER,
                               PARTIAL_HOLD_REWARD=SimpleTradingEnvTraining.PARTIAL_HOLD_REWARD,
                               dataset_id="5896df8aa22c41a3ade34d747bc9ed9a"))

        env = EnvironmentWrapper(EnvCNN, data)
        env.create(max_episode_timesteps=max(len(tck) for tck in env.data))

        agent = CustomAgent(env)
        agent.create()
        # agent.load(MlflowAPI(run_id="c3aaa7c52b3f41afb256c4c3ad4376f4").get_artifact_path())
        agent.train(episodes=100, episode_progress_indicator=env.len_data)
        agent.save()

        # pred = agent.predict()
        # log_file(pred, "pred.pkl")
