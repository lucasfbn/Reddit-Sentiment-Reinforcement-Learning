import pandas as pd
import wandb
from ray import tune

from rl.portfolio.train.envs.env import EnvCNNExtended
from rl.portfolio.train.envs.utils.reward_handler import RewardHandler
from rl.portfolio.train.networks.multi_input import Network
from rl.portfolio.train.train import train, load_data
from utils.wandb_utils import log_file

data = load_data(0, 0)


def objective(trial):
    global data

    env = EnvCNNExtended(data)
    env.NEG_REWARD = trial["NEG_REWARD"]

    with wandb.init(project="Trendstuff", group="RL Portfolio Tune Rewards 3", job_type="runs") as run:
        wandb.tensorboard.patch(save=False)
        wandb.log(trial)

        tracked_data = train(data, env, num_steps=1000000, run_dir=run.dir,
                             network=Network, features_extractor_kwargs=dict(features_dim=128))

        # log_file(tracked_data, "tracking.pkl", run)

        last_n_episodes = pd.concat(e.to_df(include_last=False) for e in tracked_data[:-10])
        reward = last_n_episodes["reward"].median()
        wandb.log({"reward_n": reward})
        tune.report(reward)


if __name__ == "__main__":
    trial = {
        "NEG_REWARD": tune.grid_search([0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10])
    }

    analysis = tune.run(
        objective,
        config=trial,
        mode="max",
        num_samples=1,
        resources_per_trial={"cpu": 2}
    )
