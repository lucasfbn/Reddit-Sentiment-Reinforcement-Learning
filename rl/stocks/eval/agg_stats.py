import pandas as pd


class AggStats:

    def __init__(self, df: pd.DataFrame):
        self.df = df.select_dtypes(include="number")

    @staticmethod
    def q(n):
        def percentile_(x):
            return x.quantile(n)

        percentile_.__name__ = str(n)
        return percentile_

    @staticmethod
    def pos(series):
        return sum(series > 0) / len(series)

    @staticmethod
    def even(series):
        return sum(series == 0) / len(series)

    @staticmethod
    def neg(series):
        return sum(series < 0) / len(series)

    @staticmethod
    def mean_neg(series):
        return series[series < 0].mean()

    @staticmethod
    def mean_pos(series):
        return series[series > 0].mean()

    @staticmethod
    def _agg_to_json(agg):
        return pd.json_normalize(agg.drop(columns=["func"]).to_dict(),
                                 sep="_").to_dict(orient="records")[0]

    def agg(self):
        agg = self.df.agg(["count", "mean", "min", "max", self.q(0.25), self.q(0.50), self.q(0.75),
                           self.q(0.9), self.q(0.95), self.pos, self.even, self.neg, self.mean_neg, self.mean_pos])
        agg = agg.round(4)
        agg["func"] = agg.index

        cols = agg.columns.tolist()
        agg = agg[cols[-1:] + cols[:-1]]
        return agg, self._agg_to_json(agg)
