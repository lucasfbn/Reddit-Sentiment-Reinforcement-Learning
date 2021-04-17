import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorforce import Runner, Agent, Environment
from learning_tensorforce.env import EnvNN

import paths
import pickle as pkl
import mlflow


def save_agent(agent, path=None):
    if path is None:
        artifact_uri = mlflow.get_artifact_uri()
        artifact_uri = "C:" + artifact_uri.split(":")[2]
        path = artifact_uri + "/model"
    agent.save(directory=path, format='numpy')


def main(data):
    EnvNN.data = data
    environment = Environment.create(environment=EnvNN)

    agent = Agent.create(
        agent='ppo', environment=environment,
        memory=2000, batch_size=32, exploration=0.01
    )

    runner = Runner(agent=agent, environment=environment)
    runner.run(num_episodes=2000)
    runner.close()

    save_agent(agent)

    agent.close()
    environment.close()


if __name__ == '__main__':
    with open(paths.datasets_data_path / "_0" / "timeseries.pkl", "rb") as f:
        data = pkl.load(f)

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Testing")  #
    mlflow.start_run()
    main(data)

    mlflow.end_run()
