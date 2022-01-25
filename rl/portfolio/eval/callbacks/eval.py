import wandb

from rl.portfolio.eval.env import EvalEnv
from rl.utils.callbacks.base import Callback


class EvalCallback(Callback):

    def __init__(self, episodes_log_interval, skip_for_first_n_intervals, data,
                 data_iter_cls, state_handler_cls, trading_env_cls):
        super().__init__(episodes_log_interval)
        self.skip_for_first_n_intervals = skip_for_first_n_intervals
        self.data = data
        self.trading_env_cls = trading_env_cls
        self.state_handler_cls = state_handler_cls
        self.data_iter_cls = data_iter_cls

        self._interval_counter = 0

    def per_episode_interval(self):

        if self._interval_counter <= self.skip_for_first_n_intervals:
            self._interval_counter += 1
        else:
            data_iter = self.data_iter_cls(self.data).sequence_iter()
            eval_env = EvalEnv(self.model, data_iter, self.state_handler_cls, self.trading_env_cls)
            profit = eval_env.eval()
            print(profit)
            wandb.log(dict(eval_profit=profit))
