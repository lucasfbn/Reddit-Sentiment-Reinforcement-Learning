from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from preprocessing.sequences import Sequence


class StateHandler(ABC):

    def __init__(self, extend):
        self.extend = extend

    @abstractmethod
    def get_state(self, sequence: Sequence) -> pd.DataFrame:
        pass

    @abstractmethod
    def shape_state(self, state: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def extend_state(self, state: pd.DataFrame, constant: Union[float, int, None]) -> pd.DataFrame:
        pass

    def forward(self, sequence: Sequence, constant: Union[float, int, None] = None):
        state = self.get_state(sequence)
        if self.extend:
            if constant is None:
                raise ValueError("Unable to extend with constant=None. Please specify constant.")
            state = self.extend_state(state, constant)
        state = self.shape_state(state)
        return state


class StateHandlerCNN(StateHandler):

    def get_state(self, sequence: Sequence):
        return sequence.arr

    def shape_state(self, state):
        state = state.values.reshape((1, state.shape[0], state.shape[1]))
        state = np.asarray(state).astype('float32')
        return state

    def extend_state(self, state, constant):
        state_columns = list(state.columns)
        constant_list = [constant for _ in range(len(state_columns))]
        new_state_row = pd.DataFrame([constant_list], columns=state_columns)
        return state.append(new_state_row)


class StateHandlerNN(StateHandler):

    def get_state(self, sequence):
        return sequence.flat

    def shape_state(self, state):
        state = np.asarray(state).astype("float32")
        state = state.reshape((state.shape[1],))
        return state

    def extend_state(self, state: pd.DataFrame, constant):
        raise NotImplementedError
