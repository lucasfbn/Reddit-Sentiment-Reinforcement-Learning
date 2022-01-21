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
    RewardHandler.TOTAL_EPISODE_END_REWARD = trial["TOTAL_EPISODE_END_REWARD"]
    RewardHandler.COMPLETED_STEPS_MAX_REWARD = trial["COMPLETED_STEPS_MAX_REWARD"]

    env = EnvCNNExtended(data)

    with wandb.init(project="Trendstuff", group="RL Portfolio Tune Rewards 1", job_type="runs") as run:
        wandb.tensorboard.patch(save=False)
        wandb.log(trial)

        tracked_data = train(data, env, num_steps=1000000, run_dir=run.dir,
                             network=Network, features_extractor_kwargs=dict(features_dim=128))

        # log_file(tracked_data, "tracking.pkl", run)

        last_n_episodes = pd.concat(e.to_df(include_last=False) for e in tracked_data[:-10])
        reward = last_n_episodes["reward"].median()
        reward_completed_steps = last_n_episodes["reward_completed_steps"].median()
        reward_discount_n_trades_left = last_n_episodes["reward_discount_n_trades_left"].median()
        wandb.log({"reward_n": reward,
                   "reward_completed_steps_n": reward_completed_steps,
                   "reward_discount_n_trades_left_n": reward_discount_n_trades_left})
        tune.report(reward)


if __name__ == "__main__":
    trial = {
        "TOTAL_EPISODE_END_REWARD": tune.grid_search(list(range(5, 11))),
        "COMPLETED_STEPS_MAX_REWARD": tune.grid_search(list(range(1, 6)))
    }

    analysis = tune.run(
        objective,
        config=trial,
        mode="max",
        num_samples=25,
        resources_per_trial={"cpu": 5}
    )

    with wandb.init(project="Trendstuff", group="RL Portfolio Tune Rewards 1", job_type="overview") as run:
        run.log({"overview_rl_portfolio_tune_rewards": wandb.Table(dataframe=analysis.results_df)})
