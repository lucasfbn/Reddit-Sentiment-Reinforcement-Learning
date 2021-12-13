from mlflow_utils import init_mlflow, load_file
from stable_baselines3 import PPO

import rl.portfolio.train.envs.pre_process.handle_sequences as hs
from rl.portfolio.train.envs.env import EnvCNNExtended
from rl.portfolio.train.envs.pre_process.merge_ticker import merge_ticker
from rl.stocks.train.networks.cnn_1d import CustomCNN
from utils.paths import mlflow_dir


def main(data_run_id, eval_run_id):
    init_mlflow(mlflow_dir, "Tests")

    data = load_file(run_id=data_run_id, fn="ticker.pkl", experiment="Datasets")
    evl = load_file(run_id=eval_run_id, fn="evl_ticker.pkl", experiment="Eval_Stocks")

    merged = merge_ticker(data, evl)
    all_sequences = hs.get_all_sequences(merged)
    all_sequences = hs.remove_invalid_sequences(all_sequences)
    sorted_buys = hs.order_sequences_daywise(all_sequences)

    total_timesteps_p_episode = len(sorted_buys)

    episodes = 10

    env = EnvCNNExtended(sorted_buys)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=32)
    )

    model = PPO('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(episodes * total_timesteps_p_episode + 1)


main(data_run_id="0643613545e44e75b8017b9973598fb4", eval_run_id="f384f58217114433875eda44495272ad")
