import pandas as pd


class Stats:

    def __init__(self, dataset):
        self.dataset = dataset

    def make_df(self):
        raise NotImplementedError

    def agg(self):
        raise NotImplementedError

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
    def _agg_to_json(agg, prefix=""):
        dict_ = pd.json_normalize(agg.drop(columns=["func"]).to_dict(),
                                  sep="_").to_dict(orient="records")[0]
        return {f'{prefix}_{k}': v for k, v in dict_.items()}

    def postprocess_agg(self, agg):
        agg = agg.round(4)
        agg["func"] = agg.index

        cols = agg.columns.tolist()
        agg = agg[cols[-1:] + cols[:-1]]
        return agg


class StatsTickerWise(Stats):

    def make_df(self):
        df = pd.DataFrame([obj.evl.to_dict() for obj in self.dataset])
        reward_df = pd.DataFrame({"reward": df["reward"]})
        positions_df = pd.DataFrame({"open_positions": df["open_positions"]})

        return reward_df, positions_df

    def agg(self):
        reward_df, positions_df = self.make_df()

        reward_agg = self.postprocess_agg(reward_df.agg(["mean", "min", "max", self.q(0.25), self.q(0.50), self.q(0.75),
                                                         self.q(0.9), self.q(0.95)]))
        positions_agg = self.postprocess_agg(positions_df.agg(["count"]))

        return self._agg_to_json(reward_agg, "ticker") | self._agg_to_json(positions_agg, "ticker")


class StatsSequenceWise(Stats):

    def make_df(self):
        df = pd.concat([obj.sequences.to_df()[["reward"]] for obj in self.dataset])
        return df.select_dtypes(include="number")

    def agg(self):
        df = self.make_df()

        agg = df.agg([self.pos, self.even, self.neg])
        agg = self.postprocess_agg(agg)
        return self._agg_to_json(agg, "seq")


class AggregatedStats(Stats):

    def make_df(self):
        pass

    def agg(self):
        return StatsTickerWise(self.dataset).agg() | StatsSequenceWise(self.dataset).agg()
