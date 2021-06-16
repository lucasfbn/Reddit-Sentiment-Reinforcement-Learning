import pandas as pd
from pandas.testing import assert_frame_equal

from preprocessing.sequences import *


def test_sequence_without_availability():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7]})

    seq = SequenceGenerator(df, sequence_len=3, include_available_days_only=False)

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

    seq = SequenceGenerator(df, sequence_len=4, include_available_days_only=False)

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
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, True, True, True, True, True],
                       "price": list(range(10, 17)), "tradeable": [True] * 7})
    seq = SequenceGenerator(df, sequence_len=3, include_available_days_only=True)
    seq.slice_sequences()
    seq.sliced_to_sequence_obj()
    result = seq.filter_availability()

    expected = [
        pd.DataFrame({"dummy": [1, 2, 3], "available": [False, False, True],
                      "price": [10, 11, 12], "tradeable": [True] * 3}),
        pd.DataFrame({"dummy": [2, 3, 4], "available": [False, True, True],
                      "price": [11, 12, 13], "tradeable": [True] * 3}),
        pd.DataFrame({"dummy": [3, 4, 5], "available": [True, True, True],
                      "price": [12, 13, 14], "tradeable": [True] * 3}),
        pd.DataFrame({"dummy": [4, 5, 6], "available": [True, True, True],
                      "price": [13, 14, 15], "tradeable": [True] * 3}),
        pd.DataFrame({"dummy": [5, 6, 7], "available": [True, True, True],
                      "price": [14, 15, 16], "tradeable": [True] * 3}),
    ]

    for r, e in zip(result, expected):
        assert_frame_equal(r.df.reset_index(drop=True), e)

    # More not available dummies
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, False, True, True, True, True]})
    seq = SequenceGenerator(df, sequence_len=3, include_available_days_only=True)
    seq.slice_sequences()
    result = seq.filter_availability()

    expected = [
        pd.DataFrame({"dummy": [2, 3, 4], "available": [False, False, True],
                      "price": [11, 12, 13], "tradeable": [True] * 3}),
        pd.DataFrame({"dummy": [3, 4, 5], "available": [False, True, True],
                      "price": [12, 13, 14], "tradeable": [True] * 3}),
        pd.DataFrame({"dummy": [4, 5, 6], "available": [True, True, True],
                      "price": [13, 14, 15], "tradeable": [True] * 3}),
        pd.DataFrame({"dummy": [5, 6, 7], "available": [True, True, True],
                      "price": [14, 15, 16], "tradeable": [True] * 3}),
    ]

    for r, e in zip(result, expected):
        assert_frame_equal(r.df.reset_index(drop=True), e.df)


def test_flat_sequence():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, True, True, True, True, True],
                       "price": list(range(10, 17)), "tradeable": [True] * 7})

    fs = FlatSequenceGenerator(df=df, sequence_len=3, include_available_days_only=False, exclude_cols_from_sequence=[
        "tradeable", "available", "price"])
    result = fs.make_sequence()

    expected = [
        pd.DataFrame({"dummy/0": [1], "dummy/1": [2], "dummy/2": [3]}),
        pd.DataFrame({"dummy/1": [2], "dummy/2": [3], "dummy/3": [4]}),
        pd.DataFrame({"dummy/2": [3], "dummy/3": [4], "dummy/4": [5]}),
        pd.DataFrame({"dummy/3": [4], "dummy/4": [5], "dummy/5": [6]}),
        pd.DataFrame({"dummy/4": [5], "dummy/5": [6], "dummy/6": [7]}),
    ]

    for r, e in zip(result, expected):
        r.df.columns = e.columns  # r uses multi-level index, e doesn't (doesn't matter for the comparison tho)
        assert_frame_equal(r.df, e, check_column_type=False)


def test_arr_sequence():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, True, True, True, True, True],
                       "price": list(range(10, 17)), "tradeable": [True] * 7})

    seq = ArraySequenceGenerator(df=df, sequence_len=3, include_available_days_only=False, exclude_cols_from_sequence=[
        "tradeable", "available", "price"])
    result = seq.make_sequence()

    expected = [
        pd.DataFrame({"dummy": [1, 2, 3]}),
        pd.DataFrame({"dummy": [2, 3, 4]}),
        pd.DataFrame({"dummy": [3, 4, 5]}),
        pd.DataFrame({"dummy": [4, 5, 6]}),
        pd.DataFrame({"dummy": [5, 6, 7]}),
    ]

    for r, e in zip(result, expected):
        assert_frame_equal(r.df.reset_index(drop=True), e)


def test_sequence_len_too_long():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7]})

    seq = SequenceGenerator(df, sequence_len=8, include_available_days_only=False)

    result = seq.slice_sequences()
    assert len(result) == 0


def test_exclude_cols():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, True, True, True, True, True],
                       "price": list(range(10, 17)), "tradeable": [True] * 7})
    seq = SequenceGenerator(df, sequence_len=3, include_available_days_only=True,
                            exclude_cols_from_sequence=["tradeable", "available", "price"])
    seq.slice_sequences()
    seq.sliced_to_sequence_obj()
    seq.filter_availability()
    result = seq.exclude_columns()

    expected = [
        pd.DataFrame({"dummy": [1, 2, 3]}),
        pd.DataFrame({"dummy": [2, 3, 4]}),
        pd.DataFrame({"dummy": [3, 4, 5]}),
        pd.DataFrame({"dummy": [4, 5, 6]}),
        pd.DataFrame({"dummy": [5, 6, 7]}),
    ]

    for r, e in zip(result, expected):
        assert_frame_equal(r.df.reset_index(drop=True), e)


def test_flat_sequence_column_order():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, True, True, True, True, True],
                       "price": [10, 11, 12, 13, 14, 15, 16], "tradeable": [True] * 7})

    fs = FlatSequenceGenerator(df=df, sequence_len=3, include_available_days_only=False,
                               exclude_cols_from_sequence=["available", "tradeable"], last_column="dummy")
    result = fs.make_sequence()

    expected = [
        pd.DataFrame(
            {"price/0": [10], "price/1": [11], "price/2": [12], "dummy/0": [1], "dummy/1": [2], "dummy/2": [3]}),
        pd.DataFrame(
            {"price/1": [11], "price/2": [12], "price/3": [13], "dummy/1": [2], "dummy/2": [3], "dummy/3": [4]}),
        pd.DataFrame(
            {"price/2": [12], "price/3": [13], "price/4": [14], "dummy/2": [3], "dummy/3": [4], "dummy/4": [5]}),
        pd.DataFrame(
            {"price/3": [13], "price/4": [14], "price/5": [15], "dummy/3": [4], "dummy/4": [5], "dummy/5": [6]}),
        pd.DataFrame(
            {"price/4": [14], "price/5": [15], "price/6": [16], "dummy/4": [5], "dummy/5": [6], "dummy/6": [7]}),
    ]

    for r, e in zip(result, expected):
        r.df.columns = e.columns  # r uses multi-level index, e doesn't (doesn't matter for the comparison tho)
        assert_frame_equal(r.df, e, check_column_type=False)

    fs = FlatSequenceGenerator(df=df, sequence_len=3, include_available_days_only=False,
                               exclude_cols_from_sequence=["available", "tradeable"], last_column="price")
    result = fs.make_sequence()

    expected = [
        pd.DataFrame(
            {"dummy/0": [1], "dummy/1": [2], "dummy/2": [3], "price/0": [10], "price/1": [11], "price/2": [12]}),
        pd.DataFrame(
            {"dummy/1": [2], "dummy/2": [3], "dummy/3": [4], "price/1": [11], "price/2": [12], "price/3": [13]}),
        pd.DataFrame(
            {"dummy/2": [3], "dummy/3": [4], "dummy/4": [5], "price/2": [12], "price/3": [13], "price/4": [14]}),
        pd.DataFrame(
            {"dummy/3": [4], "dummy/4": [5], "dummy/5": [6], "price/3": [13], "price/4": [14], "price/5": [15]}),
        pd.DataFrame(
            {"dummy/4": [5], "dummy/5": [6], "dummy/6": [7], "price/4": [14], "price/5": [15], "price/6": [16]}),
    ]

    for r, e in zip(result, expected):
        r.df.columns = e.columns  # r uses multi-level index, e doesn't (doesn't matter for the comparison tho)
        assert_frame_equal(r.df, e, check_column_type=False)


def test_empty_sequences():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False] * 7,
                       "price": [10, 11, 12, 13, 14, 15, 16], "tradeable": [True] * 7})
    seq = SequenceGenerator(df, sequence_len=3, include_available_days_only=True)
    seq.slice_sequences()
    seq.sliced_to_sequence_obj()
    result = seq.filter_availability()

    assert not result


def test_attributes():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, True, True, True, True, True],
                       "price": [10, 11, 12, 13, 14, 15, 16],
                       "tradeable": [False, False, True, True, False, False, False]})
    fs = FlatSequenceGenerator(df=df, sequence_len=3, include_available_days_only=False)
    result = fs.make_sequence()

    expected_price = [12, 13, 14, 15, 16]
    expected_tradeable = [True, True, False, False, False]
    expected_available = [True, True, True, True, True]

    i = 0
    while i < len(result):
        assert result[i].price == expected_price[i]
        assert result[i].tradeable == expected_tradeable[i]
        assert result[i].available == expected_available[i]
        i += 1
