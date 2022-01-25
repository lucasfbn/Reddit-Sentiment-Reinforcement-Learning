import pandas as pd

from pandas.testing import assert_frame_equal

from preprocessing.sequence_generator import SequenceGenerator
from dataset_handler.classes.sequence import Sequence, Metadata


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
    seq = SequenceGenerator(df, sequence_len=3, include_available_days_only=True, price_column="price")
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
        assert_frame_equal(r.data.df.reset_index(drop=True), e)

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
        assert_frame_equal(r.data.df.reset_index(drop=True), e.df)


def test_flat_sequence():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, True, True, True, True, True],
                       "price": list(range(10, 17)), "tradeable": [True] * 7})

    fs = SequenceGenerator(df=df, sequence_len=3, include_available_days_only=False, exclude_cols_from_sequence=[
        "tradeable", "available", "price"], price_column="price")
    fs.make_sequence()
    result = fs.add_flat_sequences()

    expected = [
        pd.DataFrame({"dummy/0": [1], "dummy/1": [2], "dummy/2": [3]}),
        pd.DataFrame({"dummy/1": [2], "dummy/2": [3], "dummy/3": [4]}),
        pd.DataFrame({"dummy/2": [3], "dummy/3": [4], "dummy/4": [5]}),
        pd.DataFrame({"dummy/3": [4], "dummy/4": [5], "dummy/5": [6]}),
        pd.DataFrame({"dummy/4": [5], "dummy/5": [6], "dummy/6": [7]}),
    ]

    for r, e in zip(result, expected):
        r = r.data.flat
        r.columns = e.columns  # r uses multi-level index, e doesn't (doesn't matter for the comparison tho)
        assert_frame_equal(r, e, check_column_type=False)


def test_arr_sequence():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, True, True, True, True, True],
                       "price": list(range(10, 17)), "tradeable": [True] * 7})

    seq = SequenceGenerator(df=df, sequence_len=3, include_available_days_only=False, exclude_cols_from_sequence=[
        "tradeable", "available", "price"], price_column="price")
    seq.make_sequence()
    result = seq.add_array_sequences()

    expected = [
        pd.DataFrame({"dummy": [1, 2, 3]}).transpose(),
        pd.DataFrame({"dummy": [2, 3, 4]}).transpose(),
        pd.DataFrame({"dummy": [3, 4, 5]}).transpose(),
        pd.DataFrame({"dummy": [4, 5, 6]}).transpose(),
        pd.DataFrame({"dummy": [5, 6, 7]}).transpose(),
    ]

    for r, e in zip(result, expected):
        assert_frame_equal(r.data.arr.reset_index(drop=True), e.reset_index(drop=True))


def test_sequence_len_too_long():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7]})

    seq = SequenceGenerator(df, sequence_len=8, include_available_days_only=False, price_column="price")

    result = seq.slice_sequences()
    assert len(result) == 0


def test_exclude_cols():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, True, True, True, True, True],
                       "price": list(range(10, 17)), "tradeable": [True] * 7})
    seq = SequenceGenerator(df, sequence_len=3, include_available_days_only=True,
                            exclude_cols_from_sequence=["tradeable", "available", "price"], price_column="price")
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
        assert_frame_equal(r.data.df.reset_index(drop=True), e)


def test_flat_sequence_column_order():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, True, True, True, True, True],
                       "price": [10, 11, 12, 13, 14, 15, 16], "tradeable": [True] * 7})

    fs = SequenceGenerator(df=df, sequence_len=3, include_available_days_only=False,
                           exclude_cols_from_sequence=["available", "tradeable"], price_column="dummy")
    fs.make_sequence()
    result = fs.add_flat_sequences()

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
        r = r.data.flat
        r.columns = e.columns  # r uses multi-level index, e doesn't (doesn't matter for the comparison tho)
        assert_frame_equal(r, e, check_column_type=False)

    fs = SequenceGenerator(df=df, sequence_len=3, include_available_days_only=False,
                           exclude_cols_from_sequence=["available", "tradeable"], price_column="price")
    fs.make_sequence()
    result = fs.add_flat_sequences()

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
        r = r.data.flat
        r.columns = e.columns  # r uses multi-level index, e doesn't (doesn't matter for the comparison tho)
        assert_frame_equal(r, e, check_column_type=False)


def test_empty_sequences():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False] * 7,
                       "price": [10, 11, 12, 13, 14, 15, 16], "tradeable": [True] * 7})
    seq = SequenceGenerator(df, sequence_len=3, include_available_days_only=True, price_column="price")
    seq.slice_sequences()
    seq.sliced_to_sequence_obj()
    result = seq.filter_availability()

    assert not result


def test_attributes():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5, 6, 7], "available": [False, False, True, True, True, True, True],
                       "price": [10, 11, 12, 13, 14, 15, 16],
                       "tradeable": [False, False, True, True, False, False, False],
                       "sentiment_data_available": [False, False, True, True, True, True, False],
                       "date_day_shifted": [pd.Period("01-01-2021"), pd.Period("02-01-2021"), pd.Period("03-01-2021"),
                                            pd.Period("04-01-2021"), pd.Period("05-01-2021"), pd.Period("06-01-2021"),
                                            pd.Period("07-01-2021")]})
    fs = SequenceGenerator(df=df, sequence_len=3, include_available_days_only=False, price_column="price")
    result = fs.make_sequence()

    expected_price = [12, 13, 14, 15, 16]
    expected_tradeable = [True, True, False, False, False]
    expected_available = [True, True, True, True, True]
    expected_sentiment_data_available = [True, True, True, True, False]
    expected_date = [pd.Period("03-01-2021"), pd.Period("04-01-2021"), pd.Period("05-01-2021"), pd.Period("06-01-2021"),
                     pd.Period("07-01-2021")]

    i = 0
    while i < len(result):
        assert result[i].metadata.price == expected_price[i]
        assert result[i].metadata.tradeable == expected_tradeable[i]
        assert result[i].metadata.available == expected_available[i]
        assert result[i].metadata.sentiment_data_available == expected_sentiment_data_available[i]
        assert result[i].metadata.date == expected_date[i]
        i += 1


def test_sort_sequences():
    sequences = [Sequence(metadata=Metadata(False, False, False, None, 1, None, None)),
                 Sequence(metadata=Metadata(False, False, False, None, 2, None, None)),
                 Sequence(metadata=Metadata(False, False, False, None, 3, None, None)),
                 Sequence(metadata=Metadata(False, False, False, None, 4, None, None))]

    sequences.sort(key=lambda x: x.metadata.price, reverse=False)
    for i, seq in enumerate(sequences):
        assert seq.metadata.price == i + 1
