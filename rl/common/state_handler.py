from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
import pandas as pd

from dataset_handler.classes.sequence import Sequence


class StateHandler(ABC):

    @abstractmethod
    def get_state(self, sequence: Sequence) -> pd.DataFrame:
        pass

    def forward(self, sequence: Sequence, constant: List[Union[float, int, None]] = None):
        seq_state = self.get_state(sequence)
        cons_state = np.array(constant)

        return {"timeseries": seq_state, "constants": cons_state}


class StateHandlerCNN(StateHandler):

    def get_state(self, sequence: Sequence):
        return sequence.data.arr


class StateHandlerNN(StateHandler):

    def get_state(self, sequence):
        return sequence.data.flat
