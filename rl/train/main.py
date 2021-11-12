import mlflow
from mlflow_utils import load_file, init_mlflow, setup_logger
from stable_baselines3 import PPO

import utils.paths
from rl.train.envs.env import EnvCNNExtended
from rl.train.envs.sub_envs.trading import SimpleTradingEnvTraining

if __name__ == '__main__':
    init_mlflow(utils.paths.mlflow_dir, "Tests")

    with mlflow.start_run():
        setup_logger("INFO")
        data = load_file(run_id="5896df8aa22c41a3ade34d747bc9ed9a", fn="ticker.pkl", experiment="Datasets")

        SimpleTradingEnvTraining.ENABLE_TRANSACTION_COSTS = True
        SimpleTradingEnvTraining.ENABLE_NEG_BUY_REWARD = True
        SimpleTradingEnvTraining.ENABLE_POS_SELL_REWARD = True
        SimpleTradingEnvTraining.PARTIAL_HOLD_REWARD = False
        SimpleTradingEnvTraining.HOLD_REWARD_MULTIPLIER = 0.1

        mlflow.log_params(dict(ENABLE_TRANSACTION_COSTS=SimpleTradingEnvTraining.ENABLE_TRANSACTION_COSTS,
                               ENABLE_NEG_BUY_REWARD=SimpleTradingEnvTraining.ENABLE_NEG_BUY_REWARD,
                               ENABLE_POS_SELL_REWARD=SimpleTradingEnvTraining.ENABLE_POS_SELL_REWARD,
                               TRANSACTION_FEE_BID=SimpleTradingEnvTraining.TRANSACTION_FEE_BID,
                               TRANSACTION_FEE_ASK=SimpleTradingEnvTraining.TRANSACTION_FEE_ASK,
                               HOLD_REWARD_MULTIPLIER=SimpleTradingEnvTraining.HOLD_REWARD_MULTIPLIER,
                               PARTIAL_HOLD_REWARD=SimpleTradingEnvTraining.PARTIAL_HOLD_REWARD,
                               dataset_id="5896df8aa22c41a3ade34d747bc9ed9a"))
        # callback = EveryNTimesteps(n_steps=max(len(tck) for tck in data)
        env = EnvCNNExtended(data)
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(250000)
