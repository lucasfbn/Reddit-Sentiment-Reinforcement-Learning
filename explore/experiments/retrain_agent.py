import mlflow

from utils import paths
from rl.agent import RLAgent
from rl.train.envs.env import EnvCNN
from utils.mlflow_api import load_file, MlflowAPI
from utils.util_funcs import log

"""
Examines whether we can just continue training of an already trained agent.
Issues: #77
"""

log.setLevel("INFO")

mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("Exp: Retrain agent")

n_episodes = list(range(2, 11))

for ne in n_episodes:
    log.info(f"Processing n_episodes: {ne}")

    with mlflow.start_run():
        data = load_file(run_id="f4bdae299f694599ba91c7dd1f77c9b5", fn="ticker.pkl", experiment="Datasets")

        # Train
        rla = RLAgent(environment=EnvCNN, ticker=data)

        if ne == 1:
            rla.train(n_full_episodes=1)
        else:
            rla.load_agent(MlflowAPI(run_id="e0c8640572204e4f941e30bf6437c86b").get_artifact_path())
            rla.train(n_full_episodes=1)

        rla.close()

        mlflow.log_param("Trained episodes", ne)

# os.system("shutdown /s /t 60")
