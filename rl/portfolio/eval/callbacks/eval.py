import wandb

from stable_baselines3.common.callbacks import EveryNTimesteps, BaseCallback
from rl.portfolio.eval.env import EvalEnv
from rl.common.callbacks.base import Callback


class EvalCallback(BaseCallback):
    def __init__(self, data, data_iter_cls, state_handler_cls, trading_env_cls):
        super().__init__()
        self.data = data
        self.trading_env_cls = trading_env_cls
        self.state_handler_cls = state_handler_cls
        self.data_iter_cls = data_iter_cls

    def _on_step(self):
        data_iter = self.data_iter_cls(self.data).sequence_iter()
        eval_env = EvalEnv(self.model, data_iter, self.state_handler_cls, self.trading_env_cls)
        eval_metrics = eval_env.eval()
        wandb.log(eval_metrics | dict(global_step=self.num_timesteps))


class EvalCallbackEpisodes(Callback):

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
            eval_metrics = eval_env.eval()
            wandb.log(eval_metrics | dict(global_step=self.num_timesteps))
