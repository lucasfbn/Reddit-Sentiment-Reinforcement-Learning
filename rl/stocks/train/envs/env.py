from abc import ABC, abstractmethod

import numpy as np
from gym import Env, spaces

from preprocessing.sequence import Sequence
from rl.stocks.train.envs.sub_envs.trading import SimpleTradingEnvTraining
from rl.stocks.train.envs.utils.data_iterator import DataIterator
from rl.utils.state_handler import StateHandlerCNN, StateHandlerNN


class BaseEnv(Env, ABC):

    def __init__(self, ticker):
        super().__init__()

        self.data_iter = DataIterator(ticker)
        self.trading_env = SimpleTradingEnvTraining("init")

        self.action_space = spaces.Discrete(3, )

        timeseries_shape = (10, 14)
        constants_shape = (1)

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
        return self.state_handler.forward(sequence, [self.trading_env.inventory_state()])

    def step(self, actions):
        price = self.data_iter.curr_sequence.metadata.price

        # Hold
        if actions == 0:
            reward = self.trading_env.hold(price)

        # Buy
        elif actions == 1:
            reward = self.trading_env.buy(price)

        # Sell
        elif actions == 2:
            reward = self.trading_env.sell(price)

        else:
            raise ValueError("Invalid action.")

        next_sequence = self.data_iter.next_sequence()
        next_state = self.forward_state(next_sequence)

        return next_state, reward, self.data_iter.is_episode_end(), {"reward": reward}

    def close(self):
        pass

    def reset(self):
        self.data_iter.episode_end = False
        self.data_iter.episode_count += 1

        self.data_iter.next_ticker()
        next_sequence = self.data_iter.next_sequence()
        state = self.forward_state(next_sequence)
        self.trading_env = SimpleTradingEnvTraining(ticker_name=self.data_iter.curr_ticker.name)
        return state


class EnvNN(BaseEnv):
    state_handler = StateHandlerNN()


class EnvCNN(BaseEnv):
    state_handler = StateHandlerCNN()
