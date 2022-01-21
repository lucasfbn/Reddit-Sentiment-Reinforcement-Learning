import os
from pathlib import Path

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback

import rl.portfolio.train.envs.pre_process.handle_sequences as hs
from rl.utils.callbacks.tracker import TrackCallback
from rl.portfolio.train.callbacks.log import log_func
from rl.portfolio.train.envs.env import EnvCNNExtended
from rl.portfolio.train.envs.pre_process.merge_ticker import merge_ticker
from rl.portfolio.train.networks.multi_input import Network
from utils.wandb_utils import load_artefact, log_file
from rl.portfolio.train.envs.utils.reward_handler import RewardHandler


def load_data(data_version, evl_version):
    with wandb.init(project="Trendstuff", group="Throwaway") as run:
        data = load_artefact(run, "dataset.pkl", data_version, "Dataset")
        evl = load_artefact(run, "evaluated.pkl", evl_version, "Eval_Stocks")

    merged = merge_ticker(data, evl)
    all_sequences = hs.get_all_sequences(merged)
    all_sequences = hs.remove_invalid_sequences(all_sequences)

    return all_sequences


def train(data, env, run_dir, network, features_extractor_kwargs, num_steps, shutdown=False):
    total_timesteps_p_episode = len(data)

    wandb.log(dict(
        TOTAL_EPISODE_END_REWARD=RewardHandler.TOTAL_EPISODE_END_REWARD,
        COMPLETED_STEPS_MAX_REWARD=RewardHandler.COMPLETED_STEPS_MAX_REWARD,
        FORCED_EPISODE_END_PENALTY=RewardHandler.FORCED_EPISODE_END_PENALTY
    ))

    policy_kwargs = dict(
        features_extractor_class=network,
        features_extractor_kwargs=features_extractor_kwargs
    )

    checkpoint_callback = CheckpointCallback(save_freq=total_timesteps_p_episode,
                                             save_path=Path(Path(run_dir) / "models").as_posix())
    track_callback = TrackCallback(log_func=log_func())

    model = PPO('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs,
                tensorboard_log=(Path(run_dir) / "tensorboard").as_posix())
    model.learn(num_steps, callback=[WandbCallback(), checkpoint_callback, track_callback])

    if shutdown:
        os.system('shutdown -s -t 600')

    return track_callback.data


def main():
    data = load_data(0, 0)

    with wandb.init(project="Trendstuff", group="Throwaway") as run:
        wandb.tensorboard.patch(save=False)

        env = EnvCNNExtended(data)

        tracked_data = train(data, env, num_steps=1500000, run_dir=run.dir,
                             network=Network, features_extractor_kwargs=dict(features_dim=128))

        log_file(tracked_data, "tracking.pkl", run)


if __name__ == "__main__":
    main()
