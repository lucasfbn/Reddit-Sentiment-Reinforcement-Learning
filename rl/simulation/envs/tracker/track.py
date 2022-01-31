import pandas as pd


class BaseTracker:

    def __init__(self):
        self._tracked = []

    @property
    def tracked(self):
        return pd.DataFrame(self._tracked)


class EnvStateTracker(BaseTracker):

    def __init__(self, env):
        super().__init__()

        self.env = env

    def track(self, date):
        self._tracked.append(dict(
            date=date,
            inventory_len=len(self.env.inventory),
            balance=self.env.balance
        ))


class TradeTracker(BaseTracker):

    def track(self, date, success, sequence):
        self._tracked.append(dict(
            ticker=sequence.metadata.ticker_name,
            date=date,
            success=success,
            action=sequence.evl.action,
            exec=sequence.portfolio.execute,
            tradeable=sequence.metadata.tradeable,
            price=sequence.metadata.price_raw
        ))


class Tracker:

    def __init__(self, trading_env):
        self.env_state = EnvStateTracker(trading_env)
        self.trades = TradeTracker()

    def track(self, date, success, sequence):
        self.env_state.track(date)
        self.trades.track(date, success, sequence)

    @property
    def tracked(self):
        return pd.concat([self.trades.tracked, self.env_state.tracked], axis=1)
