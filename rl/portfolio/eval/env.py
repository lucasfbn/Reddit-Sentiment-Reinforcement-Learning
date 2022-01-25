from pathlib import Path

import wandb
from stable_baselines3 import PPO

from rl.utils.predict_proba import predict_proba


class EvalEnv:

    def __init__(self, model, data_iter, state_handler_cls, trading_env_cls):
        self._data_iter = data_iter
        self._state_handler_cls = state_handler_cls
        self._trading_env_cls = trading_env_cls
        self._model = model

    @staticmethod
    def _get_variable_input(seq, env):
        inventory_state = env.inventory.inventory_state(seq)
        probability = seq.evl.buy_proba
        n_trades_left = env.n_trades_left_scaled
        trades_exhausted = env.trades_exhausted()
        return inventory_state, probability, n_trades_left, trades_exhausted

    def eval(self):
        state_handler = self._state_handler_cls()
        env = self._trading_env_cls()

        for seq, episode_end, new_date in self._data_iter:

            if new_date:
                env.new_day()

            state = state_handler.forward(seq, [*self._get_variable_input(seq, env)])
            action, _ = self._model.predict(state, deterministic=True)
            action, probas = predict_proba(self._model, state)

            env.step(action, seq)

            seq.portfolio.execute = action
            seq.portfolio.proba_execute = probas[1]

            if episode_end:
                break

        return env.n_trades / env.N_START_TRADES


if __name__ == '__main__':
    from rl.portfolio.train.envs.sub_envs.trading import TradingSimulator
    from rl.utils.state_handler import StateHandlerCNN
    from rl.portfolio.train import load_data
    from rl.portfolio.train.envs.utils.data_iterator import DataIterator

    data = load_data("2d2742q1", 0)

    data = sorted(data, key=lambda seq: seq.metadata.date)

    data_iter = DataIterator(data).sequence_iter()
    #
    run_id = "9cxzb2vg"
    model_fn = "rl_model_374448_steps.zip"

    model_path = Path(wandb.restore(name=f"models/{model_fn}",
                                    run_path=f"lucasfbn/Trendstuff/{run_id}").name)
    model = PPO.load(model_path)
    env = EvalEnv(model, data_iter, StateHandlerCNN, TradingSimulator)
    print(env.eval())