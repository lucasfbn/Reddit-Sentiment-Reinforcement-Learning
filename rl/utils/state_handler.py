from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from preprocessing.sequence import Sequence


class StateHandler(ABC):

    def __init__(self, extend):
        self.extend = extend

    @abstractmethod
    def get_state(self, sequence: Sequence) -> pd.DataFrame:
        pass

    @abstractmethod
    def extend_state(self, state: pd.DataFrame, constant: Union[float, int, None]) -> pd.DataFrame:
        pass

    def shape_state(self, state: pd.DataFrame) -> np.ndarray:
        return np.asarray(state).astype('float32')

    def cat_forward(self, sequence: Sequence, constant: list[Union[float, int, None]] = None):
        seq_state = self.get_state(sequence)
        seq_state = self.shape_state(seq_state)
        cons_state = np.array(constant)

        return {"timeseries": seq_state, "constants": cons_state}


class StateHandlerCNN(StateHandler):

    def get_state(self, sequence: Sequence):
        return sequence.data.arr

    def extend_state(self, state, constant):
        state_columns = list(state.columns)
        constant_list = [constant for _ in range(len(state_columns))]
        new_state_row = pd.DataFrame([constant_list], columns=state_columns)
        return state.append(new_state_row)


class StateHandlerNN(StateHandler):

    def get_state(self, sequence):
        return sequence.data.flat

    def extend_state(self, state: pd.DataFrame, constant):
        raise NotImplementedError
