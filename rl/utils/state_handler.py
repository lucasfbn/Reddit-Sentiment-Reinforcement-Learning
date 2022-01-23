from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from preprocessing.sequence import Sequence


class StateHandler(ABC):

    @abstractmethod
    def get_state(self, sequence: Sequence) -> pd.DataFrame:
        pass

    def shape_state(self, state: pd.DataFrame) -> np.ndarray:
        return np.asarray(state).astype('float32')

    def forward(self, sequence: Sequence, constant: list[Union[float, int, None]] = None):
        seq_state = self.get_state(sequence)
        seq_state = self.shape_state(seq_state)
        cons_state = np.array(constant)

        return {"timeseries": seq_state, "constants": cons_state}


class StateHandlerCNN(StateHandler):

    def get_state(self, sequence: Sequence):
        return sequence.data.arr


class StateHandlerNN(StateHandler):

    def get_state(self, sequence):
        return sequence.data.flat
