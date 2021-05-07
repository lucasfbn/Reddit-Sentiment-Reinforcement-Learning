import multiprocessing
from itertools import product

import mlflow
from tqdm import tqdm

import paths
from evaluate.eval_portfolio import EvaluatePortfolio
from utils import log

log.setLevel("ERROR")
mlflow.set_tracking_uri(paths.mlflow_path)
mlflow.set_experiment("Evaluating")


class CrossValidateEvaluation:

    def __init__(self, data, settings={"hold": False, "buy": True, "sell": False},
                 step_size=0.01, n_worker=10):

        self.n_worker = n_worker
        self.data = data
        self.step_size = step_size
        self.settings = settings

        self._combinations = self._get_combinations()
        self._dates_trades_combinations = None

        self._results = None

    def _get_combinations(self):

        n_true = 0
        for action, value in self.settings.items():
            if value is True:
                n_true += 1

        rng = range(0, 1 * 100, int(self.step_size * 100))
        rng = [step / 100 for step in rng]

        if n_true == 0:
            raise ValueError
        else:
            args = [rng for n in range(1, n_true + 1)]
            product_tuple = list(product(*args))

        combinations_list = []
        for tpl in product_tuple:
            combinations = {"hold": None, "buy": None, "sell": None}
            i = 0
            for action, value in self.settings.items():
                if value is True:
                    combinations[action] = tpl[i]
                    i += 1
            combinations_list.append(combinations)

        return combinations_list

    def _multi_cv(self, args):

        combinations_list = args[0]
        manager = args[1]

        dates_trades_combinations = None

        for combination in tqdm(combinations_list):
            ep = EvaluatePortfolio(eval_data=self.data, quantiles_thresholds=combination)
            ep._dates_trades_combination = self._dates_trades_combinations

            ep.initialize()
            ep.act()
            ep.force_sell()

            result = ep.get_result()
            result.update({"combination": combination})
            manager.append(result)

    @staticmethod
    def _chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _calculate_initial_dates_trades_combination(self):
        ep = EvaluatePortfolio(eval_data=self.data, quantiles_thresholds={"hold": None, "buy": None, "sell": None})
        ep.initialize()
        self._dates_trades_combinations = ep._dates_trades_combination

    def get_top_results(self, n_best):
        sorted_ = sorted(self._results, key=lambda k: k['profit'], reverse=True)
        return sorted_[:n_best]

    def log_top_results(self, n_best):
        results = self.get_top_results(n_best)
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

    def cross_validate(self):
        combinations = self._get_combinations()
        chunks = list(self._chunks(combinations, self.n_worker))
        self._calculate_initial_dates_trades_combination()

        manager = multiprocessing.Manager()
        manager = manager.list()

        with multiprocessing.Pool(self.n_worker) as p:
            p.map(self._multi_cv, product(chunks, [manager]))

        self._results = list(manager)


if __name__ == "__main__":
    import pickle as pkl

    path = "C:/Users/lucas/OneDrive/Backup/Projects/Trendstuff/storage/mlflow/mlruns/5/" \
           "472e633695ce4beab58634b5e73d10c2/artifacts/eval_test_0.pkl"
    with open(path, "rb") as f:
        data = pkl.load(f)

    # data = data[:10]

    # TODO Has the capacity to cross validate several true actions. This takes time (~45 mins  @ 100% CPU) and won't
    # work with mlflow properly. You will, therefore, have to add a global variable to filter and
    # not log everything to mlflow.
    cv_settings = {"hold": False, "buy": True, "sell": False}

    with mlflow.start_run():
        cv = CrossValidateEvaluation(data=data, settings=cv_settings, step_size=0.01, n_worker=10)
        cv.cross_validate()
        cv.log_top_results(3)
