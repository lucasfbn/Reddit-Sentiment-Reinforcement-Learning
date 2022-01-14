import mlflow
from mlflow_utils import init_mlflow, load_file, artifact_path, log_file
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

import rl.portfolio.train.envs.pre_process.handle_sequences as hs
from rl.portfolio.train.envs.env import EnvCNNExtended
from rl.portfolio.train.envs.pre_process.merge_ticker import merge_ticker
from rl.portfolio.train.networks.multi_input import Network
from utils.paths import mlflow_dir
from rl.portfolio.train.callbacks.tracker import TrackCallback


def main(data_run_id, eval_run_id):
    init_mlflow(mlflow_dir, "Training_Portfolio")

    with mlflow.start_run():
        # setup_logger("INFO")

        data = load_file(run_id=data_run_id, fn="ticker.pkl", experiment="Datasets")
        evl = load_file(run_id=eval_run_id, fn="evl_ticker.pkl", experiment="Eval_Stocks")

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
                                                 save_path=(artifact_path() / "models").as_posix())
        track_callback = TrackCallback()

        model = PPO('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs,
                    tensorboard_log=(artifact_path() / "tensorboard").as_posix())
        # model.learn(episodes * total_timesteps_p_episode + 1, callback=[checkpoint_callback])
        model.learn(4000000, callback=[checkpoint_callback, track_callback])

        log_file(fn="tracking_data", file=track_callback.data)

        # import os
        #
        # os.system('shutdown -s')


main(data_run_id="0643613545e44e75b8017b9973598fb4", eval_run_id="f384f58217114433875eda44495272ad")
