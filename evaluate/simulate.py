import pickle as pkl

import learning
import paths
from evaluate.eval_all_trades import EvaluateAllTrades


class Simulator:

    def __init__(self, model_path):
        self.model_path = model_path

        self.act_data = None

    def _load(self, path):
        with open(path, "rb") as f:
            data = pkl.load(f)
        return data

    def act(self, data_path):
        self.act_data = learning.main.main(data_path, eval=True,
                                           model_path=self.model_path, eval_out_path="eval.pkl")

    def evaluate(self, evaluate_in=None, evaluate_out=None):
        if evaluate_in is not None:
            data = self._load(evaluate_in)
        else:
            data = self.act_data

        et = EvaluateAllTrades(data, evaluate_out)
        print(et.overall_statistics())




s = Simulator(model_path=paths.models_path / "18_44---08_02-21.mdl")
# s.act(paths.test_path / "data_timeseries.pkl")
s.evaluate(paths.models_path / "18_44---08_02-21.mdl" / "eval.pkl")
