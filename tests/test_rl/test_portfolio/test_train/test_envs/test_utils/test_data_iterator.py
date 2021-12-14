from tests.utils import MockObj
from rl.portfolio.train.envs.utils.data_iterator import DataIterator
import pandas as pd

sequences = [
    MockObj(metadata=MockObj(date=pd.Period("2012-04-01"), id=0)),
    MockObj(metadata=MockObj(date=pd.Period("2012-04-02"), id=1)),
    MockObj(metadata=MockObj(date=pd.Period("2012-04-03"), id=2)),
    MockObj(metadata=MockObj(date=pd.Period("2012-04-03"), id=3)),
]


def test_basecase():
    di = DataIterator(sequences)

    expected = [
        (sequences[0], False, False),
        (sequences[1], False, True),
        (sequences[2], False, True),
        (sequences[3], True, False),
        (sequences[0], False, False),
        (sequences[1], False, True),
        (sequences[2], False, True),
        (sequences[3], True, False),
    ]

    sequence_iter = di.sequence_iter()

    for item in expected:
        res = next(sequence_iter)
        assert res == item
