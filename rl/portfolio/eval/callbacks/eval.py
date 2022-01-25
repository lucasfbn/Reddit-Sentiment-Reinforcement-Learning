import wandb

from rl.utils.callbacks.base import Callback
from rl.portfolio.eval.env import EvalEnv


class EvalCallback(Callback):

    def __init__(self, episodes_log_interval, data, data_iter_cls, state_handler_cls, trading_env_cls):
        super().__init__(episodes_log_interval)
        self.data = data
        self.trading_env_cls = trading_env_cls
        self.state_handler_cls = state_handler_cls
        self.data_iter_cls = data_iter_cls

    def per_episode_interval(self):
        data_iter = self.data_iter_cls(self.data).sequence_iter()
        eval_env = EvalEnv(self.model, data_iter, self.state_handler_cls, self.trading_env_cls)
        profit = eval_env.eval()
        wandb.log(dict(eval_profit=profit))
