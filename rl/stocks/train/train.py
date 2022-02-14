import wandb

from dataset_handler.stock_dataset import StockDatasetWandb
from rl.common.runner.train import TrainRunner
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
        ]


def main():
    data = load_data(0)

    with wandb.init(project="Trendstuff", group="Throwaway") as run:
        wandb.tensorboard.patch(save=False)

        env = EnvCNN(data)
        num_steps = data.stats.total_sequences() * 15

        runner = StockTrainRunner(run.dir, data, env)
        runner.train(Network, dict(features_dim=128), num_steps)


if __name__ == '__main__':
    main()
