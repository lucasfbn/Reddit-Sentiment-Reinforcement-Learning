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

    expected = [
        (sequences[0], False, False),
        (sequences[1], False, True),
        (sequences[2], True, True),
        (sequences[0], False, False),
        (sequences[1], False, True),
        (sequences[2], True, True),
    ]

    sequence_iter = di.sequence_iter()

    for i in range(6):
        res = next(sequence_iter)
        assert res == expected[i]
