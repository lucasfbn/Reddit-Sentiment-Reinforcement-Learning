import logging
from abc import ABC, abstractmethod
from random import randrange
from random import shuffle

import numpy as np
from gym import Env, spaces

from preprocessing.sequence import Sequence
from rl.portfolio.train.envs.sub_envs.trading import TradingSimulator
from rl.portfolio.train.envs.utils.data_iterator import DataIterator
from rl.portfolio.train.envs.utils.reward_handler import RewardHandler
from rl.utils.state_handler import StateHandlerCNN, StateHandlerNN

log = logging.getLogger("root")


class BaseEnv(Env, ABC):
    NEG_REWARD = 5

    def __init__(self, base_sequences):
        super().__init__()

        self._sequences = base_sequences

        self._data_iter = DataIterator(self._sequences)
        self._curr_state_iter = self._data_iter.sequence_iter()
        self._next_state_iter = self._data_iter.sequence_iter()

        self._trading_env = TradingSimulator()

        self.action_space = spaces.Discrete(2, )

        timeseries_shape = (10, 14)
        constants_shape = (3)

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
        inventory_state = self._trading_env.inventory.inventory_state(sequence)
        probability = sequence.evl.buy_proba
        n_trades_left = self._trading_env.n_trades_left_scaled
        return self.state_handler.cat_forward(sequence, [inventory_state, probability, n_trades_left])

    def step(self, actions):
        seq, episode_end, new_date = next(self._curr_state_iter)

        if new_date:
            self._trading_env.new_day()

        reward, success = self._trading_env.step(actions, seq)

        reward_handler = RewardHandler()
        reward = reward_handler.discount_cash_bound(reward, seq.evl.days_cash_bound)

        if self._trading_env.trades_exhausted():
            reward -= self.NEG_REWARD

        next_sequence, _, _ = next(self._next_state_iter)

        next_state = self.forward_state(next_sequence)

        self._data_iter.step()

        return next_state, reward, episode_end, {"reward": reward,
                                                 "episode_end": episode_end,
                                                 "new_date": new_date,
                                                 "n_trades_left": self._trading_env.n_trades_left_scaled,
                                                 "trades_exhausted": self._trading_env.trades_exhausted(),
                                                 "completed_steps": self._data_iter.perc_completed_steps,
                                                 "total_steps": len(self._data_iter.sequences),
                                                 "current_steps": self._data_iter.steps}

    def close(self):
        pass

    def _shuffle_sequences(self):
        shuffle(self._sequences)
        self._sequences = sorted(self._sequences, key=lambda seq: seq.metadata.date)
        start_index = randrange(0, len(self._sequences))
        episode_sequences = self._sequences[start_index:]
        return episode_sequences, len(episode_sequences)

    def reset(self):
        sequences, n_max_episodes = self._shuffle_sequences()
        self._data_iter = DataIterator(sequences)
        self._curr_state_iter = self._data_iter.sequence_iter()
        self._next_state_iter = self._data_iter.sequence_iter()

        next_sequence, _, _ = next(self._next_state_iter)
        state = self.forward_state(next_sequence)
        self._trading_env = TradingSimulator()
        return state


class EnvNN(BaseEnv):
    state_handler = StateHandlerNN(extend=False)


class EnvNNExtended(EnvNN):
    state_handler = StateHandlerNN(extend=True)


class EnvCNN(BaseEnv):
    state_handler = StateHandlerCNN(extend=False)


class EnvCNNExtended(EnvCNN):
    state_handler = StateHandlerCNN(extend=True)


if __name__ == "__main__":
    import pickle as pkl
    from mlflow_utils import base_logger

    base_logger("DEBUG")

    with open("temp.pkl", "rb") as f:
        data = pkl.load(f)

    env = EnvCNNExtended(data)

    state = env.reset()

    for _ in range(100):
        next_state, reward, episode_end, _ = env.step(env.action_space.sample())

    env.close()
