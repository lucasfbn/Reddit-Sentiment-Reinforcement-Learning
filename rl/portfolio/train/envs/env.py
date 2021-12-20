import logging
from abc import ABC, abstractmethod
from random import randrange
from random import shuffle

import numpy as np
from gym import Env, spaces

from preprocessing.sequence import Sequence
from rl.portfolio.train.envs.sub_envs.trading import TradingSimulator
from rl.portfolio.train.envs.utils.data_iterator import DataIterator
from rl.portfolio.train.envs.utils.forced_episode_reward_handler import ForcedEpisodeRewardHandler
from rl.utils.state_handler import StateHandlerCNN, StateHandlerNN

log = logging.getLogger("root")


class BaseEnv(Env, ABC):

    def __init__(self, base_sequences):
        super().__init__()

        self._sequences = base_sequences

        self._data_iter = DataIterator(self._sequences)
        self._curr_state_iter = self._data_iter.sequence_iter()
        self._next_state_iter = self._data_iter.sequence_iter()

        self._forced_episode_end_handler: ForcedEpisodeRewardHandler = None

        self._trading_env = TradingSimulator()

        self.action_space = spaces.Discrete(2, )
        shape = self._get_initial_observation_state_shape()
        self.observation_space = spaces.Box(low=np.zeros(shape),
                                            high=np.ones(shape),
                                            dtype=np.float64)

    @property
    @abstractmethod
    def state_handler(self):
        pass

    def forward_state(self, sequence: Sequence):
        inventory_state = self._trading_env.inventory.inventory_state(sequence)
        probability = sequence.evl.buy_proba
        n_trades_left = self._trading_env.n_trades_left_scaled
        return self.state_handler.forward(sequence, [inventory_state, probability, n_trades_left])

    def _get_first_sequence(self):
        return self._data_iter.sequences[0]

    def _get_initial_observation_state_shape(self):
        return self.forward_state(self._get_first_sequence()).shape

    def _check_forced_episode_end(self, total_episode_end, intermediate_episode_end, reward):

        if total_episode_end:  # regular episode end
            episode_end = True
            reward += 25
        else:
            episode_end = intermediate_episode_end

            if episode_end:  # e.g. forced episode end
                neg_reward = self._forced_episode_end_handler.get_episode_end_reward(self._data_iter.episode)
                reward -= neg_reward
                log.debug(f"Forced episode end. Reduced reward by {neg_reward}. "
                          f"Percentage of completed episodes: {self._data_iter.perc_completed_episodes}")

        return episode_end, reward

    def step(self, actions):
        seq, episode_end, new_date = next(self._curr_state_iter)

        if new_date:
            self._trading_env.new_day()

        reward = self._trading_env.step(actions, seq)
        intermediate_episode_end = self._trading_env.trades_exhausted()

        episode_end, reward = self._check_forced_episode_end(episode_end,
                                                             intermediate_episode_end,
                                                             reward)

        next_sequence, _, _ = next(self._next_state_iter)

        next_state = self.forward_state(next_sequence)

        self._data_iter.increment_episode()

        return next_state, reward, episode_end, {}

    def close(self):
        pass

    def _shuffle_sequences(self):
        shuffle(self._sequences)
        self._sequences = sorted(self._sequences, key=lambda seq: seq.metadata.date)
        start_index = randrange(0, len(self._sequences))
        episode_sequences = self._sequences[start_index:]
        return episode_sequences, start_index, len(episode_sequences)

    def reset(self):
        sequences, start_index, end_index = self._shuffle_sequences()
        self._data_iter = DataIterator(sequences)
        self._curr_state_iter = self._data_iter.sequence_iter()
        self._next_state_iter = self._data_iter.sequence_iter()

        self._forced_episode_end_handler = ForcedEpisodeRewardHandler(start_index, end_index)

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
