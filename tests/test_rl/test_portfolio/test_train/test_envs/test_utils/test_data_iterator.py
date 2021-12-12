from tests.utils import MockObj
from rl.portfolio.train.envs.utils.data_iterator import DataIterator
import pandas as pd

sequences = [
    MockObj(metadata=MockObj(date=pd.Period("2012-04-01"))),
    MockObj(metadata=MockObj(date=pd.Period("2012-04-02"))),
    MockObj(metadata=MockObj(date=pd.Period("2012-04-03"))),
]


def test_basecase():
    di = DataIterator(sequences)

    for _ in range(2):
        di.episode_end = False
        assert di.next_sequence() == sequences[0]
        assert di.is_new_date() == False
        assert di.next_sequence() == sequences[1]
        assert di.is_new_date() == False
        assert di.next_sequence() == sequences[2]
        assert di.is_new_date() == False and di.episode_end  # new_date() == False because of episode end
