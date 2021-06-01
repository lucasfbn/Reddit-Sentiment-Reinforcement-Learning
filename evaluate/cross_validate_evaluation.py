from dataclasses import dataclass
from itertools import product
from evaluate.eval_portfolio import EvaluatePortfolio
from utils import log
from tqdm import tqdm
import multiprocessing
import copy
import mlflow
import paths


@dataclass
class Interval:
    start: float
    end: float
    step: float

    def values(self):
        rng = range(int(self.start * 100), int(self.end * 101), int(self.step * 100))
        rng = [step / 100 for step in rng]
        return rng


@dataclass
class Choice:
    choices: list

    def values(self):
        return self.choices


class ParameterTuning:

    def __init__(self, data, parameter, n_worker=10):
        self.n_worker = n_worker
        self.parameter = parameter
        self.data = data

        self._combinations = None
        self._combinations_mapped = []

    def _generate_combinations(self):
        values = []
        for param, input in self.parameter.items():
            values.append(input.values())

        self._combinations = list(product(*values))

    def _reassign_combinations(self):

        keys = self.parameter.keys()

        for combination in self._combinations:

            combination_key_dict = {}

            for k, c in zip(keys, combination):
                combination_key_dict[k] = c

            self._combinations_mapped.append(combination_key_dict)

    def _correct_mapping(self):

        for mapping in self._combinations_mapped:
            keys = mapping.keys()
            base = {"hold": None, "buy": None, "sell": None}
            base_keys = base.keys()

            for key in keys:
                if key in base_keys:
                    base[key] = mapping[key]

            for key in base_keys:
                mapping.pop(key, None)

            mapping["quantiles_thresholds"] = base

    @staticmethod
    def _chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _save_init_values(self):
        log.info("Retrieving initial values...")
        ep = EvaluatePortfolio(data=self.data, quantiles_thresholds={"hold": None, "buy": None, "sell": None})
        ep.initialize()
        self._dates_trades_combinations = ep._dates_trades_combination
        self._min_date, self._max_date = ep._min_date, ep._max_date

    def _multi_cross_validation(self, args):

        combinations_list = args[0]
        manager = args[1]

        dates_trades_combinations = None

        for combination in tqdm(combinations_list):
            ep = EvaluatePortfolio(data=self.data, **combination)
            ep._dates_trades_combination = self._dates_trades_combinations
            ep._min_date, ep._max_date = self._min_date, self._max_date

            ep.initialize()
            ep.act()
            ep.force_sell()

            result = ep.get_result()
            result.update({"combination": combination})
            manager.append(result)

    def _run_multiprocess(self):
        chunks = list(self._chunks(self._combinations_mapped, self.n_worker))

        manager = multiprocessing.Manager()
        manager = manager.list()

        with multiprocessing.Pool(self.n_worker) as p:
            p.map(self._multi_cross_validation, product(chunks, [manager]))

        self._results = list(manager)

    def get_top_results(self, n_best):
        sorted_ = sorted(self._results, key=lambda k: k['profit'], reverse=True)
        return sorted_[:n_best]

    def log_top_results(self, n_best):
        results = copy.deepcopy(self.get_top_results(n_best))
        assert len(results) == n_best

        metrics = {}
        for n, r in enumerate(results):
            metrics[f"profit_{n}"] = r["profit"]
            metrics[f"balance_{n}"] = r["balance"]
        mlflow.log_metrics(metrics)

        for r in results:
            with mlflow.start_run(nested=True):
                mlflow.log_params(r)

        # Pop profit and balance from first dict in list. All dicts should have the same params.
        results[0].pop("profit")
        results[0].pop("balance")

        mlflow.log_params(results[0])

    def tune(self):
        self._generate_combinations()
        self._reassign_combinations()
        self._correct_mapping()

        log.info(f"Number of total combinations:  {len(self._combinations_mapped)}, "
                 f"Estimated runtime: {len(self._combinations_mapped) * 13 / 60 / 60 / self.n_worker} h")

        self._save_init_values()

        self._run_multiprocess()


if __name__ == "__main__":
    path = "C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/5/30ed88b45c974e768caf2949651edfb6/artifacts/eval_train.pkl"

    import pickle as pkl

    with open(path, "rb") as f:
        data = pkl.load(f)
    # print(Interval(0, 1, 0.01).values())
    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Evaluating")
    with mlflow.start_run():
        pt = ParameterTuning(data,
                             parameter={"buy": Interval(0.7, 1, 0.01)},
                                        # "max_trades_per_day": Choice([1, 3, 5, 7, 10, 15]),
                                        # "max_price_per_stock": Choice([10, 20, 25, 30, 40, 50]),
                                        # "max_investment_per_trade": Choice([0.03, 0.05, 0.07, 0.1])},
                             n_worker=10)
        pt.tune()
        print(pt.get_top_results(3))
        # pt.log_top_results(3)
