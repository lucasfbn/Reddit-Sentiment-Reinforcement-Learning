from tqdm import tqdm

from rl.common.predict_proba import predict_proba


class EvalEnv:

    def __init__(self, dataset, model, state_handler_cls, trading_env_cls):
        self.trading_env_cls = trading_env_cls
        self.state_handler = state_handler_cls()
        self.model = model
        self.dataset = dataset

    def _eval_sequences(self, sequences):

        trading_env = self.trading_env_cls()

        for seq in sequences:
            state = self.state_handler.forward(seq,
                                               [trading_env.inventory_state()])
            action, proba = predict_proba(self.model, state)

            if action == 0:
                reward = trading_env.hold(seq.metadata.price)
            elif action == 1:
                reward = trading_env.buy(seq.metadata.price)
            elif action == 2:
                reward = trading_env.sell(seq.metadata.price)
            else:
                raise ValueError("Invalid action.")

            seq.evl.action = action
            seq.evl.reward = reward
            seq.evl.open_positions = len(trading_env.inventory)
            seq.evl.split_probas(proba)

        return len(trading_env.inventory)

    def eval(self):
        for ticker in tqdm(self.dataset):
            ticker.evl.open_positions = self._eval_sequences(ticker.sequences)
            ticker.evl.reward = ticker.sequences.aggregated_rewards()
            ticker.sequences.backtrack()

        return self.dataset


if __name__ == '__main__':
    from pathlib import Path

    import wandb
    from stable_baselines3 import PPO

    from utils.wandb_utils import log_to_summary
    from rl.stocks.train.train import load_data
    from rl.common.state_handler import StateHandlerCNN
    from rl.stocks.train.envs.sub_envs.trading import SimpleTradingEnv
    from rl.stocks.eval.callbacks.stats import AggregatedStats

    dataset = load_data(0)[:10]

    run_id = "1j3fhfbn"
    model_fn = "rl_model_10000_steps.zip"

    model_path = Path(wandb.restore(name=f"models/{model_fn}",
                                    run_path=f"lucasfbn/Trendstuff/{run_id}").name)
    model = PPO.load(model_path)

    env = EvalEnv(dataset, model, StateHandlerCNN, SimpleTradingEnv)
    evl_result = env.eval()

    agg_stats = AggregatedStats(evl_result).agg()

    with wandb.init(project="Trendstuff", group="Throwaway") as run:
        log_to_summary(run, agg_stats)
        dataset.log_as_file(run)
