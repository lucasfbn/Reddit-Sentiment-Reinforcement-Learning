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

    def track(self, date, success, action_pair):
        operation = action_pair["operation"]
        action = action_pair["action"]
        proba = action_pair["proba"]

        self._tracked.append(dict(
            ticker=operation.ticker,
            date=date,
            success=success,
            action=action,
            proba=proba,
            tradeable=operation.tradeable,
            price=operation.price
        ))


class Tracker:

    def __init__(self, trading_env):
        self.env_state = EnvStateTracker(trading_env)
        self.trades = TradeTracker()

    def track(self, date, success, action_pair):
        self.env_state.track(date)
        self.trades.track(date, success, action_pair)

    @property
    def tracked(self):
        return pd.concat([self.trades.tracked, self.env_state.tracked], axis=1)
