from pathlib import Path

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (CheckpointCallback,
                                                EveryNTimesteps)

from rl.stocks.train.callbacks.callbacks import EpisodeEndCallback
from rl.stocks.train.envs.env import EnvCNNExtended
from rl.stocks.train.envs.sub_envs.trading import SimpleTradingEnvTraining
from rl.stocks.train.networks.cnn_1d_tune import Networks, TuneableNetwork
from utils.wandb_utils import load_file

"""
https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html
https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
"""

if __name__ == '__main__':
    with wandb.init(project="Trendstuff", group="RL_Stocks") as run:
        data = load_file(run, "dataset.pkl", 0, "Dataset")

        wandb.log(dict(ENABLE_TRANSACTION_COSTS=SimpleTradingEnvTraining.ENABLE_TRANSACTION_COSTS,
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

        episodes = 10

        env = EnvCNNExtended(data)

        networks = Networks(in_channels=env.observation_space.shape[0]).networks

        for network, features_dim in networks:
            log_callback = EveryNTimesteps(n_steps=total_timesteps_p_episode, callback=EpisodeEndCallback())
            checkpoint_callback = CheckpointCallback(save_freq=total_timesteps_p_episode,
                                                     save_path=Path(Path(run.dir) / "models").as_posix())

            policy_kwargs = dict(
                features_extractor_class=TuneableNetwork,
                features_extractor_kwargs=dict(cnn=network, features_dim=features_dim)
            )

            model = PPO('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
            model.learn(episodes * total_timesteps_p_episode + 1, callback=[log_callback, checkpoint_callback])

        # import os
        #
        # os.system('shutdown -s')
