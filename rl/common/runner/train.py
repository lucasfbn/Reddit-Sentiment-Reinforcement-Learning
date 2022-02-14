import os
from pathlib import Path

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback


class TrainRunner:

    def __init__(self, run_dir, data, env, model_checkpoint: int = None, shutdown: bool = False):
        self.env = env
        self.data = data
        self.shutdown = shutdown
        self.run_dir = run_dir
        self.model_checkpoint = model_checkpoint
        self.callbacks_lst = [WandbCallback()]

        self._manage_callbacks()

    def _manage_callbacks(self):
        if self.model_checkpoint is not None:
            self.callbacks_lst.append(CheckpointCallback(save_freq=self.model_checkpoint,
                                                         save_path=Path(Path(self.run_dir) / "models").as_posix()))
        self.callbacks_lst += self.callbacks()

    def callbacks(self):
        return []

    def config(self):
        return {}

    def train(self, network, network_kwargs, num_steps):
        policy_kwargs = dict(
            features_extractor_class=network,
            features_extractor_kwargs=network_kwargs
        )

        model = PPO('MultiInputPolicy', self.env, verbose=1, policy_kwargs=policy_kwargs,
                    tensorboard_log=(Path(self.run_dir) / "tensorboard").as_posix())
        model.learn(num_steps, callback=self.callbacks_lst)

        if self.shutdown:
            os.system('shutdown -s -t 600')

        wandb.config.update(self.config())

        return model
