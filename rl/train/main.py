import mlflow
from mlflow_utils import MlflowUtils, init_mlflow, load_file, setup_logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (CheckpointCallback,
                                                EveryNTimesteps)

import utils.paths
from rl.train.callbacks.callbacks import EpisodeEndCallback
from rl.train.envs.env import EnvCNN, EnvCNNExtended
from rl.train.envs.sub_envs.trading import SimpleTradingEnvTraining
from rl.train.networks.cnn_1d import CustomCNN

"""
https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html
https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
"""

if __name__ == '__main__':
    init_mlflow(utils.paths.mlflow_dir, "Training")

    with mlflow.start_run():
        setup_logger("INFO")
        data = load_file(run_id="c3925e0cbdcd4620b5cb909e1a629419", fn="ticker.pkl", experiment="Datasets")

        mlflow.log_params(dict(ENABLE_TRANSACTION_COSTS=SimpleTradingEnvTraining.ENABLE_TRANSACTION_COSTS,
                               ENABLE_NEG_BUY_REWARD=SimpleTradingEnvTraining.ENABLE_NEG_BUY_REWARD,
                               ENABLE_POS_SELL_REWARD=SimpleTradingEnvTraining.ENABLE_POS_SELL_REWARD,
                               TRANSACTION_FEE_BID=SimpleTradingEnvTraining.TRANSACTION_FEE_BID,
                               TRANSACTION_FEE_ASK=SimpleTradingEnvTraining.TRANSACTION_FEE_ASK,
                               HOLD_REWARD_MULTIPLIER=SimpleTradingEnvTraining.HOLD_REWARD_MULTIPLIER,
                               PARTIAL_HOLD_REWARD=SimpleTradingEnvTraining.PARTIAL_HOLD_REWARD,
                               dataset_id="5896df8aa22c41a3ade34d747bc9ed9a"))

        len_sequences = [len(tck) for tck in data]
        max_timesteps = max(len_sequences)
        total_timesteps_p_episode = sum(len_sequences)

        episodes = 7

        log_callback = EveryNTimesteps(n_steps=total_timesteps_p_episode, callback=EpisodeEndCallback())
        checkpoint_callback = CheckpointCallback(save_freq=total_timesteps_p_episode,
                                                 save_path=(MlflowUtils().get_artifact_path() / "models").as_posix())

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=32)
        )

        env = EnvCNN(data)
        model = PPO('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
        model.learn(episodes * total_timesteps_p_episode + 1, callback=[log_callback, checkpoint_callback])
