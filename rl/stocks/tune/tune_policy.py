import pandas as pd
import wandb
from ray import tune

from rl.stocks.train.envs.env import EnvCNN
from rl.stocks.train.networks.multi_input import Network
from rl.stocks.train.train import train, load_data
from ray.tune.suggest.optuna import OptunaSearch


# data = load_data(0)


def objective_(trial):
    global data

    env = EnvCNN(data)

    with wandb.init(project="Trendstuff", group="RL Stock Tune Policy 1", job_type="runs") as run:
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
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([8, 16, 32, 64, 128, 256, 512]),
        "gamma": tune.choice([0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
        "ent_coef": tune.loguniform(0.00000001, 0.1),
        "clip_range": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "gae_lambda": tune.choice([0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]),
        "max_grad_norm": tune.choice([0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]),
        "vf_coef": tune.uniform(0, 1),
        "net_arch": tune.choice([64, 128, 256])
    }

    search_algorithm = OptunaSearch()

    analysis = tune.run(
        objective,
        config=trial,
        mode="max",
        num_samples=10,
        search_alg=search_algorithm,
        resources_per_trial={"cpu": 2}
    )
