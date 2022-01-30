import logging
from abc import ABC, abstractmethod
from random import randrange
from random import shuffle

import numpy as np
from gym import Env, spaces

from dataset_handler.classes.sequence import Sequence
from rl.portfolio.train.envs.sub_envs.trading import TradingSimulator
from rl.portfolio.train.envs.utils.data_iterator import DataIterator
from rl.portfolio.train.envs.utils.reward_handler import RewardHandler
from rl.utils.state_handler import StateHandlerCNN, StateHandlerNN

log = logging.getLogger("root")


class BaseEnv(Env, ABC):

    def __init__(self, base_sequences):
        super().__init__()

        self._sequences = base_sequences

        self.data_iter = DataIterator(self._sequences)
        self._curr_state_iter = self.data_iter.sequence_iter()
        self._next_state_iter = self.data_iter.sequence_iter()

        self.trading_env = TradingSimulator()

        self.action_space = spaces.Discrete(2, )

        timeseries_shape = (10, 14)
        constants_shape = (4)

        self.observation_space = spaces.Dict(
            {"timeseries": spaces.Box(low=np.zeros(timeseries_shape),
                                      high=np.ones(timeseries_shape),
                                      dtype=np.float64),
             "constants": spaces.Box(low=np.zeros(constants_shape),
                                     high=np.ones(constants_shape),
                                     dtype=np.float64)}
        )

    @property
    @abstractmethod
    def state_handler(self):
        pass

    def forward_state(self, sequence: Sequence):
        inventory_state = self.trading_env.inventory.inventory_state(sequence)
        probability = sequence.evl.buy_proba
        n_trades_left = self.trading_env.n_trades_left_scaled
        trades_exhausted = self.trading_env.trades_exhausted()
        inv_len = self.trading_env.inventory.inv_len()
        inv_ratio = self.trading_env.inventory.inv_len() / (
                self.trading_env.n_trades + self.trading_env.inventory.inv_len())
        trades_ratio = self.trading_env.n_trades / (self.trading_env.n_trades + self.trading_env.inventory.inv_len())
        return self.state_handler.forward(sequence, [inventory_state, probability, inv_ratio, trades_ratio])

    def step(self, actions):
        seq, episode_end, new_date = next(self._curr_state_iter)

        if new_date:
            self.trading_env.new_day()

        reward, success = self.trading_env.step(actions, seq)

        intermediate_episode_end = self.trading_env.trades_exhausted()

        reward_handler = RewardHandler()
        # reward = reward_handler.discount_cash_bound(reward, seq.evl.days_cash_bound)

        penalty_base = -1.0
        inv_ratio, trades_ratio = reward_handler.inv_trades_ratio(inv_len=self.trading_env.inventory.inv_len(),
                                                                  n_trades=self.trading_env.n_trades)
        factor = reward_handler.go1(trades_ratio)

        total_reward = reward + (penalty_base * factor)

        # reward_completed_steps = reward_handler.add_reward_completed_steps(reward, self.data_iter.perc_completed_steps)
        # reward_discount_n_trades_left = reward_handler.discount_n_trades_left(reward_completed_steps,
        #                                                                       self.trading_env.n_trades_left_scaled)

        # total_reward = reward_handler.penalize_forced_episode_end(reward_completed_steps,
        #                                                           intermediate_episode_end)
        # total_reward = reward_handler.reward_total_episode_end(total_reward, episode_end)
        #
        # episode_end = bool(max(int(intermediate_episode_end), int(episode_end)))

        next_sequence, _, _ = next(self._next_state_iter)

        next_state = self.forward_state(next_sequence)

        self.data_iter.step()

        return next_state, total_reward, episode_end, {"reward": reward,
                                                       "reward_discount_n_trades_left": 0,
                                                       "total_reward": total_reward,
                                                       "episode_end": episode_end,
                                                       "new_date": new_date,
                                                       "inv_ratio": inv_ratio,
                                                       "intermediate_episode_end": intermediate_episode_end,
                                                       "n_trades_left": self.trading_env.n_trades_left_scaled,
                                                       "trades_exhausted": self.trading_env.trades_exhausted(),
                                                       "completed_steps": self.data_iter.perc_completed_steps,
                                                       "total_steps": len(self.data_iter.sequences),
                                                       "current_steps": self.data_iter.steps}

    def close(self):
        pass

    def _shuffle_sequences(self):
        start_index = randrange(0, len(self._sequences))
        episode_sequences = self._sequences[start_index:]
        return episode_sequences, len(episode_sequences)

    def reset(self):
        sequences, n_max_episodes = self._shuffle_sequences()
        self.data_iter = DataIterator(sequences)
        self._curr_state_iter = self.data_iter.sequence_iter()
        self._next_state_iter = self.data_iter.sequence_iter()

        next_sequence, _, _ = next(self._next_state_iter)
        state = self.forward_state(next_sequence)
        self.trading_env = TradingSimulator()
        return state


class EnvNN(BaseEnv):
    state_handler = StateHandlerNN()


class EnvCNN(BaseEnv):
    state_handler = StateHandlerCNN()


if __name__ == "__main__":
    import pickle as pkl
    from mlflow_utils import base_logger

    base_logger("DEBUG")

    with open("temp.pkl", "rb") as f:
        data = pkl.load(f)

    env = EnvCNN(data)

    state = env.reset()

    for _ in range(100):
        next_state, reward, episode_end, _ = env.step(env.action_space.sample())

    env.close()
