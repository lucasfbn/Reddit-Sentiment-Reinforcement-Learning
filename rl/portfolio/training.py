import os
from pathlib import Path

import pandas as pd
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from wandb.integration.sb3 import WandbCallback

import rl.portfolio.train.envs.pre_process.handle_sequences as hs
from dataset_handler.stock_dataset import StockDatasetWandb
from rl.portfolio.eval.callbacks.eval import EvalCallback
from rl.portfolio.train.callbacks.log import LogCallback
from rl.portfolio.train.envs.env import EnvCNN
from rl.portfolio.train.networks.multi_input import Network
from utils.wandb_utils import log_to_summary


def load_data(meta_run_id, dataset_version):
    with wandb.init(project="Trendstuff", group="Throwaway") as run:
        dataset = StockDatasetWandb()
        dataset.wandb_load_meta_file(meta_run_id, run)
        dataset.wandb_load_data(run, dataset_version)

    all_sequences = hs.get_all_sequences(dataset)
    all_sequences = hs.remove_invalid_sequences(all_sequences)

    for seq in all_sequences:
        seq.metadata.date = pd.Period(seq.metadata.date)

    all_sequences = sorted(all_sequences, key=lambda seq: seq.metadata.date)

    return all_sequences, dataset


def train(data, env, run_dir, network, features_extractor_kwargs, num_steps,
          run_eval=True, shutdown=False, model_checkpoints=False):
    total_timesteps_p_episode = len(data)

    summary = {}

    policy_kwargs = dict(
        features_extractor_class=network,
        features_extractor_kwargs=features_extractor_kwargs
    )

    callbacks = [WandbCallback(), LogCallback(1)]
    if model_checkpoints:
        checkpoint_callback = CheckpointCallback(save_freq=25000,
                                                 save_path=Path(Path(run_dir) / "models").as_posix())
        callbacks.append(checkpoint_callback)
    if run_eval:
        eval_callback = EvalCallback(data, env.data_iter.__class__, env.state_handler.__class__,
                                     env.trading_env.__class__)

        eval_callback_event = EveryNTimesteps(n_steps=25000, callback=eval_callback)
        callbacks.append(eval_callback_event)

    model = PPO('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs,
                tensorboard_log=(Path(run_dir) / "tensorboard").as_posix())
    model.learn(num_steps, callback=callbacks)

    if shutdown:
        os.system('shutdown -s -t 600')

    return model, summary


def main():
    data, _ = load_data("2d2742q1", 0)

    with wandb.init(project="Trendstuff", group="RL Portfolio Train") as run:
        wandb.tensorboard.patch(save=False)

        env = EnvCNN(data)

        model, summary = train(data, env, num_steps=500000, run_dir=run.dir,
                               network=Network, features_extractor_kwargs=dict(features_dim=128),
                               model_checkpoints=True)

        log_to_summary(run, summary)


if __name__ == "__main__":
    main()
