from preprocessing.tasks import Ticker
from rl.train.envs.utils.data_iterator import DataIterator


def test_next_sequence():
    ticker = [Ticker(_, None) for _ in range(5)]
    for tck in ticker:
        tck.sequences = [tck.name for _ in range(5)]

    data_iter = DataIterator(ticker)

    result_curr_ticker = []
    result_curr_seq = []
    expected_curr_ticker = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]

    for i in range(12):
        data_iter.next_ticker()
        result_curr_ticker.append(data_iter.curr_ticker)
        result_curr_seq.append(data_iter.curr_sequences[0])

    result_curr_ticker = [tck.name for tck in result_curr_ticker]

    assert result_curr_ticker == expected_curr_ticker
    assert result_curr_seq == expected_curr_ticker
