import wandb

from stable_baselines3.common.callbacks import BaseCallback
from rl.stocks.eval.env import EvalEnv
from rl.stocks.eval.callbacks.stats import AggregatedStats


class EvalCallback(BaseCallback):
    def __init__(self, data, state_handler_cls, trading_env_cls):
        super().__init__()
        self.data = data
        self.state_handler_cls = state_handler_cls
        self.trading_env_cls = trading_env_cls

    def _on_step(self):
        eval_env = EvalEnv(self.data, self.model, self.state_handler_cls, self.trading_env_cls)
        self.data = eval_env.eval()

        agg_stats = AggregatedStats(self.data).agg()

        wandb.log(agg_stats | dict(global_step=self.num_timesteps))
