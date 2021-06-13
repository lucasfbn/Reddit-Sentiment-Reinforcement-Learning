import pandas as pd
from pandas.testing import assert_frame_equal

from preprocessing.sequences import FlatSequence, ArraySequence, Sequence


def test_sequence_without_availability():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7]})

    seq = Sequence(df, sequence_len=3, include_available_days_only=False)

    result = seq.slice_sequences()
    expected = [
        pd.DataFrame({"dummy": [1, 2, 3]}),
        pd.DataFrame({"dummy": [2, 3, 4]}),
        pd.DataFrame({"dummy": [3, 4, 5]}),
        pd.DataFrame({"dummy": [4, 5, 6]}),
        pd.DataFrame({"dummy": [5, 6, 7]}),
    ]

    for r, e in zip(result, expected):
        assert_frame_equal(r.reset_index(drop=True), e)

    result = seq.filter_availability()
    for r, e in zip(result, expected):
        assert_frame_equal(r.reset_index(drop=True), e)


def test_sequence_longer_sequence_len():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7]})

    seq = Sequence(df, sequence_len=4, include_available_days_only=False)

    result = seq.slice_sequences()
    expected = [
        pd.DataFrame({"dummy": [1, 2, 3, 4]}),
        pd.DataFrame({"dummy": [2, 3, 4, 5]}),
        pd.DataFrame({"dummy": [3, 4, 5, 6]}),
        pd.DataFrame({"dummy": [4, 5, 6, 7]}),
        pd.DataFrame({"dummy": [5, 6, 7, 8]}),
    ]

    for r, e in zip(result, expected):
        assert_frame_equal(r.reset_index(drop=True), e)

    result = seq.filter_availability()
    for r, e in zip(result, expected):
        assert_frame_equal(r.reset_index(drop=True), e)


def test_sequence_with_availability():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, True, True, True, True, True]})
    seq = Sequence(df, sequence_len=3, include_available_days_only=True)
    seq.slice_sequences()
    result = seq.filter_availability()

    expected = [
        pd.DataFrame({"dummy": [1, 2, 3], "available": [False, False, True]}),
        pd.DataFrame({"dummy": [2, 3, 4], "available": [False, True, True]}),
        pd.DataFrame({"dummy": [3, 4, 5], "available": [True, True, True]}),
        pd.DataFrame({"dummy": [4, 5, 6], "available": [True, True, True]}),
        pd.DataFrame({"dummy": [5, 6, 7], "available": [True, True, True]}),
    ]

    for r, e in zip(result, expected):
        assert_frame_equal(r.reset_index(drop=True), e)

    # More not available dummies
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, False, True, True, True, True]})
    seq = Sequence(df, sequence_len=3, include_available_days_only=True)
    seq.slice_sequences()
    result = seq.filter_availability()

    expected = [
        pd.DataFrame({"dummy": [2, 3, 4], "available": [False, False, True]}),
        pd.DataFrame({"dummy": [3, 4, 5], "available": [False, True, True]}),
        pd.DataFrame({"dummy": [4, 5, 6], "available": [True, True, True]}),
        pd.DataFrame({"dummy": [5, 6, 7], "available": [True, True, True]}),
    ]

    for r, e in zip(result, expected):
        assert_frame_equal(r.reset_index(drop=True), e)


def test_flat_sequence():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7]})

    fs = FlatSequence(df=df, sequence_len=3, include_available_days_only=False)
    result = fs.make_sequence()

    expected = [
        pd.DataFrame({"dummy/0": [1], "dummy/1": [2], "dummy/2": [3]}),
        pd.DataFrame({"dummy/1": [2], "dummy/2": [3], "dummy/3": [4]}),
        pd.DataFrame({"dummy/2": [3], "dummy/3": [4], "dummy/4": [5]}),
        pd.DataFrame({"dummy/3": [4], "dummy/4": [5], "dummy/5": [6]}),
        pd.DataFrame({"dummy/4": [5], "dummy/5": [6], "dummy/6": [7]}),
    ]

    for r, e in zip(result, expected):
        r.columns = e.columns  # r uses multi-level index, e doesn't (doesn't matter for the comparison tho)
        assert_frame_equal(r, e, check_column_type=False)


def test_arr_sequence():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7]})

    seq = ArraySequence(df=df, sequence_len=3, include_available_days_only=False)
    result = seq.make_sequence()

    expected = [
        pd.DataFrame({"dummy": [1, 2, 3]}),
        pd.DataFrame({"dummy": [2, 3, 4]}),
        pd.DataFrame({"dummy": [3, 4, 5]}),
        pd.DataFrame({"dummy": [4, 5, 6]}),
        pd.DataFrame({"dummy": [5, 6, 7]}),
    ]

    for r, e in zip(result, expected):
        assert_frame_equal(r.reset_index(drop=True), e)


def test_sequence_len_too_long():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7]})

    seq = Sequence(df, sequence_len=8, include_available_days_only=False)

    result = seq.slice_sequences()
    assert len(result) == 0
