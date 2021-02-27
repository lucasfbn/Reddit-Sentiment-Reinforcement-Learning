import datetime
import pickle as pkl

import tensorflow as tf

import paths
from evaluate.eval_portfolio import EvaluatePortfolio
from learning.model import deep_q_model
from utils import tracker


def main(input_path, eval=False, model_path=None):
    with open(input_path, "rb") as f:
        data = pkl.load(f)

    if eval:
        model = tf.keras.models.load_model(model_path)
        eval_data = deep_q_model(data, eval=True, model=model)

        eval_path = paths.eval_data_path / f"{datetime.datetime.now().strftime('%H:%M-%d_%m-%y')}.pkl"
        with open(eval_path, "wb") as f:
            pkl.dump(eval_data, f)
    else:
        model = deep_q_model(data, eval=False)
        model_path = paths.models_path / f"{datetime.datetime.now().strftime('%H:%M-%d_%m-%y')}"
        model.save(model_path)
        tracker.add({"model": model_path.name})

        eval_data = deep_q_model(data, eval=True, model=model)
        eval_path = paths.eval_data_path / f"{datetime.datetime.now().strftime('%H:%M-%d_%m-%y')}.pkl"
        with open(eval_path, "wb") as f:
            pkl.dump(eval_data, f)

    ep = EvaluatePortfolio(eval_data)
    ep.act()
    ep.force_sell()

    tracker.add({"dataset": input_path.parent.name,
                 "eval_path": eval_path.name,
                 "profit": ep.profit,
                 "balance": ep.balance})

    tracker.new(kind="eval")


if __name__ == "__main__":
    main(paths.datasets_data_path / "_0" / "timeseries.pkl",
         eval=False,
         model_path=None)
