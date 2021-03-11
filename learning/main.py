import datetime
import pickle as pkl

import tensorflow as tf

import paths
from evaluate.eval_portfolio import EvaluatePortfolio
from learning.agent import CNN_Agent, NN_Agent
from learning.env import Env_NN, Env_CNN
from learning.model import deep_q_model
from utils import save_config, Config

config = Config(**dict(
    data_path=paths.datasets_data_path / "_9" / "timeseries.pkl",
    kind="CNN",
    eval=True,
    model_path=paths.models_path / "21-50 04_03-21"
))


def main():
    with open(config.data_path, "rb") as f:
        data = pkl.load(f)

    if config.kind == "CNN":
        shape = data[0]["data"][0].shape
        agent = CNN_Agent(state_size=shape[0], action_size=3, feature_size=shape[1],
                          memory_len=1000, eval=eval)
        agent.build_model()
        env = Env_CNN()
    else:
        agent = NN_Agent(state_size=data[0]["data"].shape[1], action_size=3, memory_len=1000, eval=eval)
        agent.build_model()
        env = Env_NN()

    if config.eval:
        model_path = config.model_path
        model = tf.keras.models.load_model(model_path)
        eval_data = deep_q_model(data, agent=agent, env=env, eval=True, model=model)

        eval_path = paths.eval_data_path / f"{datetime.datetime.now().strftime('%H-%M %d_%m-%y')}.pkl"
        with open(eval_path, "wb") as f:
            pkl.dump(eval_data, f)
    else:
        model = deep_q_model(data, agent=agent, env=env, eval=False)
        model_path = paths.models_path / f"{datetime.datetime.now().strftime('%H-%M %d_%m-%y')}"
        model.save(model_path)

        agent.eval = True

        eval_data = deep_q_model(data, agent=agent, env=env, eval=True, model=model)
        eval_path = paths.eval_data_path / f"{datetime.datetime.now().strftime('%H-%M %d_%m-%y')}.pkl"
        with open(eval_path, "wb") as f:
            pkl.dump(eval_data, f)

    ep = EvaluatePortfolio(eval_data)
    ep.act()
    ep.force_sell()

    config.model_path = model_path
    config.eval_path = eval_path.name
    config.profit = ep.profit
    config.balance = ep.balance
    save_config(config, kind="eval")


if __name__ == "__main__":
    main()
