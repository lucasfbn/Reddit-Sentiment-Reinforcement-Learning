from pathlib import Path

import wandb
from stable_baselines3 import PPO

from rl.common.predict_proba import predict_proba


class EvalEnv:

    def __init__(self, model, data_iter, state_handler_cls, trading_env_cls):
        self._data_iter = data_iter
        self._state_handler_cls = state_handler_cls
        self._trading_env_cls = trading_env_cls
        self._model = model

    def _get_variable_input(self, seq, env):
        inventory_state = env.inventory.inventory_state(seq)
        probability = seq.evl.buy_proba
        inv_ratio, trades_ratio = self._inv_trades_ratio(env)
        return inventory_state, probability, inv_ratio, trades_ratio

    def _inv_trades_ratio(self, env):
        inv_len = env.inventory.inv_len()
        inv_ratio = inv_len / (inv_len + env.n_trades)
        trades_ratio = 1 - inv_ratio
        return inv_ratio, trades_ratio

    def eval(self):
        state_handler = self._state_handler_cls()
        env = self._trading_env_cls()

        actions = []

        for seq, episode_end, new_date in self._data_iter:

            if new_date:
                env.new_day()

            state = state_handler.forward(seq, [*self._get_variable_input(seq, env)])
            # action, _ = self._model.predict(state, deterministic=True)
            action, probas = predict_proba(self._model, state)

            env.step(action, seq)

            seq.portfolio.execute = action
            seq.portfolio.proba_execute = probas[1]

            actions.append(action)

            if episode_end:
                break

        profit = env.n_trades / env.N_START_TRADES
        exec_ratio = sum(actions) / len(actions)

        return {"eval_profit": profit, "eval_exec_ratio": exec_ratio}


if __name__ == '__main__':
    from rl.portfolio.train.envs.sub_envs.trading import TradingSimulator
    from rl.common.state_handler import StateHandlerCNN
    from rl.portfolio.training import load_data
    from rl.portfolio.train.envs.utils.data_iterator import DataIterator

    data, dataset = load_data("2d2742q1", 0)

    data_iter = DataIterator(data).sequence_iter()
    #
    run_id = "1xt7s0hf"
    model_fn = "rl_model_275000_steps.zip"

    model_path = Path(wandb.restore(name=f"models/{model_fn}",
                                    run_path=f"lucasfbn/Trendstuff/{run_id}").name)
    model = PPO.load(model_path)
    env = EvalEnv(model, data_iter, StateHandlerCNN, TradingSimulator)
    evl_result = env.eval()
    print(evl_result)

    with wandb.init(project="Trendstuff", group="RL Portfolio Eval") as run:
        dataset.log_as_file(run)
