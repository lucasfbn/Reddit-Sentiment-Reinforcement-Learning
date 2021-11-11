import mlflow
from utils import paths
from utils.mlflow_api import load_file, log_file
import pandas as pd

mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("N_Episodes_Impact_1")

run_ids = [
    "12829d4fd8fb408cbeee4d2e08f30c1f",
    "bb04288abd984b16b93af30702c8516a",
    "7ee1659cb16b48cc85d81e779f2ea908",
    "9077606aca9d4adda448a4aa0ecb244d",
    "353bf42678674979a8f95e482308a676",
    "822ca31758114397b0e66da27be01fc9",
    "2328d70a3b2f49fb87453ec3445e5e62",
    "8c5a1e2799944a5dac7d1c85bec76ea6",
]

all_eval_stats = []
all_eval_probability_stats = []

for ri in run_ids:
    with mlflow.start_run(run_id=ri):
        run = mlflow.active_run()
        trained_episodes = run.data.params["Trained episodes"]

        df = load_file(run_id=ri, fn="eval_stats.csv")
        df = df.append(pd.Series(name=f"Above: {trained_episodes} episodes"))
        df = df.append(pd.Series(name=f" "))
        all_eval_stats.append(df)

        df = load_file(run_id=ri, fn="eval_probability_stats.csv")
        df = df.append(pd.Series(name=f"Above: {trained_episodes} episodes"))
        df = df.append(pd.Series(name=f" "))
        all_eval_probability_stats.append(df)

all_eval_stats = pd.concat(all_eval_stats)
all_eval_probability_stats = pd.concat(all_eval_probability_stats)

with mlflow.start_run():
    mlflow.log_param("RESULTS", True)
    log_file(all_eval_stats, "eval_stats.csv")
    log_file(all_eval_probability_stats, "eval_probability_stats.csv")
