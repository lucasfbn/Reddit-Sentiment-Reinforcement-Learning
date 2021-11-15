import mlflow
from mlflow_utils import load_file, init_mlflow, setup_logger, log_file, MlflowUtils
from stable_baselines3 import PPO
from tqdm import tqdm

import utils.paths
from rl.train.envs.env import EnvCNNExtended


def predict(data, model_path):
    env = EnvCNNExtended(data)
    model = PPO.load(model_path, env)

    for ticker in tqdm(data):

        for sequence in ticker.sequences:
            state = env.next_state(sequence)

            action, _ = model.predict(state)
            action = action[0]
            actions_proba = {"hold": 1, "buy": 1, "sell": 1}

            sequence.add_eval(action, actions_proba)
    return data


if __name__ == "__main__":
    init_mlflow(utils.paths.mlflow_dir, "Tests")

    with mlflow.start_run():
        setup_logger("INFO")
        data = load_file(run_id="5896df8aa22c41a3ade34d747bc9ed9a", fn="ticker.pkl", experiment="Datasets")

        model_path = MlflowUtils(run_id="7400b5a610fa4b82bbfcdfe8c65e9e9b", experiment="Tests").get_artifact_path()
        model_path = model_path / "models" / "rl_model_510039_steps.zip"

        evaluated = predict(data, model_path)
