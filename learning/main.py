import datetime
import pickle as pkl

import tensorflow as tf

import paths
from evaluate.eval_portfolio import EvaluatePortfolio
from learning.agent import Agent
from learning.config import configs
from learning.env import Env_NN, Env_CNN
from learning.model import DeepQ
from utils import save_config, Config


def main(config):
    with open(config.general.data_path, "rb") as f:
        data = pkl.load(f)

    if config.general.kind == "CNN":
        shape = data[0]["data"][0].shape
        agent = Agent(state_size=shape[0], action_size=3, feature_size=shape[1],
                      gamma=config.agent.gamma, epsilon=config.agent.epsilon,
                      epsilon_decay=config.agent.epsilon_decay, epsilon_min=config.agent.epsilon_min,
                      randomness=config.agent.randomness, memory_len=config.agent.memory_len,
                      evaluate=config.general.evaluate)
        agent.build_model(config.model.name)
        env = Env_CNN()
    else:
        agent = Agent(state_size=data[0]["data"].shape[1], action_size=3,
                      gamma=config.agent.gamma, epsilon=config.agent.epsilon,
                      epsilon_decay=config.agent.epsilon_decay, epsilon_min=config.agent.epsilon_min,
                      randomness=config.agent.randomness, memory_len=config.agent.memory_len,
                      evaluate=config.general.evaluate)
        agent.build_model(config.model.name)
        env = Env_NN()

    deep_q = DeepQ(data, agent, env)

    model = None
    if not config.general.evaluate:
        model = deep_q.train(n_episodes=config.model.n_episodes, batch_size=config.model.batch_size)

        config.general.model_path = paths.models_path / f"{datetime.datetime.now().strftime('%H-%M %d_%m-%y')}"
        model.save(config.general.model_path)

    if model is None:
        assert config.general.evaluate is True
        model = tf.keras.models.load_model(config.general.model_path)

    eval_data = deep_q.evaluate(model=model)

    eval_path = paths.eval_data_path / f"{datetime.datetime.now().strftime('%H-%M %d_%m-%y')}.pkl"
    with open(eval_path, "wb") as f:
        pkl.dump(eval_data, f)

    ep = EvaluatePortfolio(eval_data)
    ep.act()
    ep.force_sell()

    config.general.data_path = config.general.data_path.parent.name
    config.general.model_path = config.general.model_path
    config.general.eval_path = eval_path.name
    config.general.profit = ep.profit
    config.general.balance = ep.balance
    save_config([config.general, config.agent, config.model], kind="eval")


if __name__ == "__main__":
    # main(config)

    for nc in configs:
        config = Config(**dict(
            general=nc[0],
            agent=nc[1],
            model=nc[2]
        ))
        main(config)
