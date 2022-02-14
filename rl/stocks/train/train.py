import wandb
from stable_baselines3.common.callbacks import EveryNTimesteps

from dataset_handler.stock_dataset import StockDatasetWandb
from rl.common.runner.train import TrainRunner
from rl.stocks.eval.callbacks.eval import EvalCallback
from rl.stocks.train.callbacks.log import LogCallback
from rl.stocks.train.envs.env import EnvCNN
from rl.stocks.train.envs.sub_envs.trading import SimpleTradingEnvTraining
from rl.stocks.train.networks.multi_input import Network


def load_data(data_version):
    with wandb.init(project="Trendstuff", group="Throwaway") as run:
        data = StockDatasetWandb()
        data.wandb_load(run, data_version)

    return data


class StockTrainRunner(TrainRunner):

    def config(self):
        return dict(ENABLE_TRANSACTION_COSTS=SimpleTradingEnvTraining.ENABLE_TRANSACTION_COSTS,
                    ENABLE_NEG_BUY_REWARD=SimpleTradingEnvTraining.ENABLE_NEG_BUY_REWARD,
                    ENABLE_POS_SELL_REWARD=SimpleTradingEnvTraining.ENABLE_POS_SELL_REWARD,
                    TRANSACTION_FEE_BID=SimpleTradingEnvTraining.TRANSACTION_FEE_BID,
                    TRANSACTION_FEE_ASK=SimpleTradingEnvTraining.TRANSACTION_FEE_ASK,
                    HOLD_REWARD_MULTIPLIER=SimpleTradingEnvTraining.HOLD_REWARD_MULTIPLIER,
                    PARTIAL_HOLD_REWARD=SimpleTradingEnvTraining.PARTIAL_HOLD_REWARD)

    def callbacks(self):
        return [
            LogCallback(episodes_log_interval=len(self.data)),
            EveryNTimesteps(n_steps=len(self.data), callback=EvalCallback(self.data,
                                                                          self.env.state_handler.__class__,
                                                                          self.env.trading_env.__class__))
        ]


def main():
    data = load_data(0)

    with wandb.init(project="Trendstuff", group="Throwaway") as run:
        wandb.tensorboard.patch(save=False)

        env = EnvCNN(data)
        total_sequences = data.stats.total_sequences()

        runner = StockTrainRunner(run.dir, data, env, model_checkpoint=total_sequences)
        runner.train(Network, dict(features_dim=128), total_sequences * 15)


if __name__ == '__main__':
    main()
