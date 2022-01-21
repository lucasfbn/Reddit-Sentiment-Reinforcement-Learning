import os
from pathlib import Path

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (CheckpointCallback,
                                                EveryNTimesteps)

from rl.stocks.train.callbacks.callbacks import EpisodeEndCallback
from rl.stocks.train.envs.env import EnvCNNExtended
from rl.stocks.train.envs.sub_envs.trading import SimpleTradingEnvTraining
from rl.stocks.train.networks.multi_input import Network
from utils.wandb_utils import load_artefact

"""
https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html
https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
"""


def load_data(data_version):
    with wandb.init(project="Trendstuff", group="Throwaway") as run:
        data = load_artefact(run, "dataset.pkl", data_version, "Dataset")

    return data


def train(data, env, run_dir, network, policy_args, features_extractor_kwargs, num_steps, shutdown=False):
    len_sequences = [len(tck) for tck in data]
    max_timesteps = max(len_sequences)
    total_timesteps_p_episode = sum(len_sequences)

    wandb.log(dict(ENABLE_TRANSACTION_COSTS=SimpleTradingEnvTraining.ENABLE_TRANSACTION_COSTS,
                   ENABLE_NEG_BUY_REWARD=SimpleTradingEnvTraining.ENABLE_NEG_BUY_REWARD,
                   ENABLE_POS_SELL_REWARD=SimpleTradingEnvTraining.ENABLE_POS_SELL_REWARD,
                   TRANSACTION_FEE_BID=SimpleTradingEnvTraining.TRANSACTION_FEE_BID,
                   TRANSACTION_FEE_ASK=SimpleTradingEnvTraining.TRANSACTION_FEE_ASK,
                   HOLD_REWARD_MULTIPLIER=SimpleTradingEnvTraining.HOLD_REWARD_MULTIPLIER,
                   PARTIAL_HOLD_REWARD=SimpleTradingEnvTraining.PARTIAL_HOLD_REWARD))

    policy_kwargs = dict(
        features_extractor_class=network,
        features_extractor_kwargs=features_extractor_kwargs
    )

    log_callback = EveryNTimesteps(n_steps=total_timesteps_p_episode, callback=EpisodeEndCallback())
    checkpoint_callback = CheckpointCallback(save_freq=total_timesteps_p_episode,
                                             save_path=Path(Path(run_dir) / "models").as_posix())

    model = PPO('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs,
                tensorboard_log=(Path(run_dir) / "tensorboard").as_posix(), **policy_args)
    model.learn(num_steps * total_timesteps_p_episode + 1, callback=[log_callback, checkpoint_callback])

    if shutdown:
        os.system('shutdown -s -t 600')


def main():
    data = load_data(0)

    with wandb.init(project="Trendstuff", group="Throwaway") as run:
        wandb.tensorboard.patch(save=False)

        env = EnvCNNExtended(data)

        train(data, env, num_steps=15, run_dir=run.dir,
              network=Network, features_extractor_kwargs=dict(features_dim=128))


if __name__ == '__main__':
    main()
