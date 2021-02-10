import learning
from learning.env import StockEnv
from learning.agent import Agent

import paths
import pickle as pkl


class Simulator:

    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

        self.evaluated_data = None

    def evaluate(self):
        self.evaluated_data = learning.main.main(self.data_path, eval=True,
                                                 model_path=self.model_path, eval_out_path="eval.pkl")
