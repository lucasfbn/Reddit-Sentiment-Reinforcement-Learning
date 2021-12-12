from tests.utils import MockObj
from rl.portfolio.train.envs.utils.data_iterator import DataIterator
import pandas as pd

sequences = [
    MockObj(metadata=MockObj(date=pd.Period("2012-04-01"))),
    MockObj(metadata=MockObj(date=pd.Period("2012-04-02"))),
]


def test_basecase():
    di = DataIterator(sequences)
    assert di.next_sequence() == (sequences[0], False)
    assert di.next_sequence() == (sequences[1], True)
    assert di.episode_end is True
