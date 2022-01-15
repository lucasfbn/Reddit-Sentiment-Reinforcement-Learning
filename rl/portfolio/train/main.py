from pathlib import Path

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback

import rl.portfolio.train.envs.pre_process.handle_sequences as hs
from rl.portfolio.train.callbacks.tracker import TrackCallback
from rl.portfolio.train.envs.env import EnvCNNExtended
from rl.portfolio.train.envs.pre_process.merge_ticker import merge_ticker
from rl.portfolio.train.networks.multi_input import Network
from utils.wandb_utils import load_artefact, log_file


def main():
    with wandb.init(project="Trendstuff", group="RL_Portfolio") as run:
        wandb.tensorboard.patch(save=False)

        data = load_artefact(run, "dataset.pkl", 0, "Dataset")
        evl = load_artefact(run, "evaluated.pkl", 0, "Eval_Stocks")

        merged = merge_ticker(data, evl)
        all_sequences = hs.get_all_sequences(merged)
        all_sequences = hs.remove_invalid_sequences(all_sequences)

        total_timesteps_p_episode = len(all_sequences)

        episodes = 1000

        env = EnvCNNExtended(all_sequences)

        policy_kwargs = dict(
            features_extractor_class=Network,
            features_extractor_kwargs=dict(features_dim=128)
        )

        checkpoint_callback = CheckpointCallback(save_freq=total_timesteps_p_episode,
                                                 save_path=Path(Path(run.dir) / "models").as_posix())
        track_callback = TrackCallback()

        model = PPO('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs,
                    tensorboard_log=(Path(run.dir) / "tensorboard").as_posix())
        # model.learn(episodes * total_timesteps_p_episode + 1, callback=[checkpoint_callback])
        model.learn(10000, callback=[WandbCallback(), checkpoint_callback, track_callback])
        log_file(track_callback.data, "tracking.pkl", run)

        # import os
        #
        # os.system('shutdown -s')


main()
