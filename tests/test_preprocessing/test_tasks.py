import pytest
from pandas import Timestamp, Period
from pandas.testing import assert_frame_equal, assert_series_equal

from preprocessing.tasks import *


def test_add_time():
    df = pd.DataFrame({"end": ["2021-05-20 05:59"]})
    result = add_time.run(df)
    expected = pd.DataFrame([{'end': '2021-05-20 05:59', date_col: Timestamp('2021-05-20 05:59:00'),
                              date_day_col: Period('2021-05-20', 'D')}])
    assert_frame_equal(result, expected)


def test_shift_time():
    df = pd.DataFrame([{date_col: Timestamp('2021-05-20 08:00:00')},
                       {date_col: Timestamp('2021-05-20 12:00:00')},
                       {date_col: Timestamp('2021-05-21 08:00:00')}])
    result = shift_time.run(df, start_hour=8, start_min=0)
    expected = pd.DataFrame([{date_col: Timestamp('2021-05-20 08:00:00'),
                              date_shifted_col: Timestamp('2021-05-21 00:00:00'),
                              date_day_shifted_col: Period('2021-05-21', 'D')},
                             {date_col: Timestamp('2021-05-20 12:00:00'),
                              date_shifted_col: Timestamp('2021-05-21 04:00:00'),
                              date_day_shifted_col: Period('2021-05-21', 'D')},
                             {date_col: Timestamp('2021-05-21 08:00:00'),
                              date_shifted_col: Timestamp('2021-05-22 00:00:00'),
                              date_day_shifted_col: Period('2021-05-22', 'D')}])
    assert_frame_equal(result, expected)


def test_scale_sentiment_data_daywise():
    df = pd.DataFrame([
        {date_day_shifted_col: Period('2021-05-21', 'D'), 'to_be_scaled': 0, "exclude": 999},
        {date_day_shifted_col: Period('2021-05-21', 'D'), 'to_be_scaled': 1, "exclude": 777},
        {date_day_shifted_col: Period('2021-05-21', 'D'), 'to_be_scaled': 2, "exclude": 888}
    ])

    # Scale without dropping cols afterwards
    result = scale_sentiment_data_daywise.run(df, sentiment_data_cols=["to_be_scaled"], drop_unscaled_cols=False)
    expected = pd.DataFrame([
        {date_day_shifted_col: Period('2021-05-21', 'D'), 'to_be_scaled': 0, "exclude": 999,
         "to_be_scaled_scaled": 0.00000},
        {date_day_shifted_col: Period('2021-05-21', 'D'), 'to_be_scaled': 1, "exclude": 777,
         "to_be_scaled_scaled": 0.50000},
        {date_day_shifted_col: Period('2021-05-21', 'D'), 'to_be_scaled': 2, "exclude": 888,
         "to_be_scaled_scaled": 1.00000}
    ])
    assert_frame_equal(result[0], expected)
    assert result[1] == ["to_be_scaled", "to_be_scaled_scaled"]

    # Scale with dropping cols afterwards
    result = scale_sentiment_data_daywise.run(df, sentiment_data_cols=["to_be_scaled"], drop_unscaled_cols=True)
    expected = pd.DataFrame([
        {date_day_shifted_col: Period('2021-05-21', 'D'), "exclude": 999,
         "to_be_scaled_scaled": 0.00000},
        {date_day_shifted_col: Period('2021-05-21', 'D'), "exclude": 777,
         "to_be_scaled_scaled": 0.50000},
        {date_day_shifted_col: Period('2021-05-21', 'D'), "exclude": 888,
         "to_be_scaled_scaled": 1.00000}
    ])
    assert_frame_equal(result[0], expected)
    assert result[1] == ["to_be_scaled_scaled"]


def test_grp_by_ticker():
    df = pd.DataFrame([{'ticker': "GME", date_shifted_col: Timestamp('2021-05-21 00:00:00'), "dummy_value": 1},
                       {'ticker': "GME", date_shifted_col: Timestamp('2021-05-20 00:00:00'), "dummy_value": 1},
                       {'ticker': "TSLA", date_shifted_col: Timestamp('2021-05-19 00:00:00'), "dummy_value": 1},
                       {'ticker': "TSLA", date_shifted_col: Timestamp('2021-05-21 00:00:00'), "dummy_value": 1},
                       {'ticker': "TSLA", date_shifted_col: Timestamp('2021-05-20 00:00:00'), "dummy_value": 1}])
    result = grp_by_ticker.run(df)

    expected_df_1 = pd.DataFrame(
        [{'ticker': "GME", date_shifted_col: Timestamp('2021-05-21 00:00:00'), "dummy_value": 1},
         {'ticker': "GME", date_shifted_col: Timestamp('2021-05-20 00:00:00'),
          "dummy_value": 1}])

    expected_df_2 = pd.DataFrame(
        [{'ticker': "TSLA", date_shifted_col: Timestamp('2021-05-19 00:00:00'), "dummy_value": 1},
         {'ticker': "TSLA", date_shifted_col: Timestamp('2021-05-21 00:00:00'), "dummy_value": 1},
         {'ticker': "TSLA", date_shifted_col: Timestamp('2021-05-20 00:00:00'), "dummy_value": 1}])

    assert result[0].name == "GME" and result[1].name == "TSLA"
    assert_frame_equal(result[0].df.reset_index(drop=True), expected_df_1)
    assert_frame_equal(result[1].df.reset_index(drop=True), expected_df_2)


def test_drop_ticker_with_too_few_data():
    df_1 = pd.DataFrame([{'ticker': "GME", date_shifted_col: Timestamp('2021-05-20 00:00:00'), "dummy_value": 1},
                         {'ticker': "GME", date_shifted_col: Timestamp('2021-05-21 00:00:00'), "dummy_value": 1}])
    ticker_1 = Ticker(name="GME", df=df_1)
    df_2 = pd.DataFrame(
        [{'ticker': "TSLA", date_shifted_col: Timestamp('2021-05-19 00:00:00'), "dummy_value": 1},
         {'ticker': "TSLA", date_shifted_col: Timestamp('2021-05-20 00:00:00'), "dummy_value": 1},
         {'ticker': "TSLA", date_shifted_col: Timestamp('2021-05-21 00:00:00'), "dummy_value": 1}])
    ticker_2 = Ticker(name="TSLA", df=df_2)

    result = drop_ticker_with_too_few_data.run([ticker_1, ticker_2], ticker_min_len=3)
    assert len(result) == 1 and result[0].name == "TSLA"


def test_mark_trainable_days():
    df = pd.DataFrame([{'ticker': "TSLA", date_shifted_col: Timestamp('2021-05-18 00:00:00')},
                       {'ticker': "TSLA", date_shifted_col: Timestamp('2021-05-19 00:00:00')},
                       {'ticker': "TSLA", date_shifted_col: Timestamp('2021-05-20 00:00:00')},
                       {'ticker': "TSLA", date_shifted_col: Timestamp('2021-05-21 00:00:00')}])
    result = mark_trainable_days.run(Ticker(name="TSLA", df=df), ticker_min_len=2)
    expected = pd.Series([False, False, True, True], name="available")
    assert_series_equal(result.df["available"], expected)


def test_sort_ticker_df_chronologically():
    df = pd.DataFrame([{date_shifted_col: Timestamp('2021-05-19 00:00:00')},
                       {date_shifted_col: Timestamp('2021-05-21 00:00:00')},
                       {date_shifted_col: Timestamp('2021-05-20 00:00:00')}])
    result = sort_ticker_df_chronologically.run(Ticker(None, df), by=[date_shifted_col])
    expected = pd.DataFrame([{date_shifted_col: Timestamp('2021-05-19 00:00:00')},
                             {date_shifted_col: Timestamp('2021-05-20 00:00:00')},
                             {date_shifted_col: Timestamp('2021-05-21 00:00:00')}])
    assert_frame_equal(result.df.reset_index(drop=True), expected.reset_index(drop=True))


def test_add_price_data():
    # For in depth test check the tests of stock_prices.py
    # We will only test whether the exclusion is done correctly when an error was raised

    # Use non existing ticker to check for MissingDataException
    df = pd.DataFrame([{'ticker': "AHJZT", date_day_col: Timestamp('2021-05-21 00:00:00')},
                       {'ticker': "AHJZT", date_day_col: Timestamp('2021-05-19 00:00:00')}])
    result = add_price_data.run(Ticker("AHJZT", df), price_data_start_offset=0, enable_live_behaviour=False)
    assert result.exclude is True

    df = pd.DataFrame([{'ticker': "AHJZT", date_day_col: Timestamp('2021-05-21 00:00:00')},
                       {'ticker': "AHJZT", date_day_col: Timestamp('2021-05-19 00:00:00')}])
    result = add_price_data.run(Ticker("AHJZT", df), price_data_start_offset=0, enable_live_behaviour=True)
    assert result.exclude is True


def test_merge_prices_with_ticker_df():
    df = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'),
                                    Period('2021-05-20', 'D')],
                       "compound": [1, 2]})

    prices = pd.DataFrame({"date_day": [
        Period('2021-05-10', 'D'),
        Period('2021-05-11', 'D'),
        Period('2021-05-12', 'D'),
        Period('2021-05-13', 'D'),
        Period('2021-05-14', 'D'),
        # Period('2021-05-15', 'D'), NO TRADE DAY
        # Period('2021-05-16', 'D'), NO TRADE DAY
        Period('2021-05-17', 'D'),
        Period('2021-05-18', 'D'),
        Period('2021-05-19', 'D'),
        Period('2021-05-20', 'D')
    ], "Close": [1, 2, 3, 4, 5, 8, 9, 10, 11]})

    expected = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'),
                                          Period('2021-05-11', 'D'),
                                          Period('2021-05-12', 'D'),
                                          Period('2021-05-13', 'D'),
                                          Period('2021-05-14', 'D'),
                                          # Period('2021-05-15', 'D'), NO TRADE DAY
                                          # Period('2021-05-16', 'D'), NO TRADE DAY
                                          Period('2021-05-17', 'D'),
                                          Period('2021-05-18', 'D'),
                                          Period('2021-05-19', 'D'),
                                          Period('2021-05-20', 'D')],
                             "compound": [1, None, None, None, None, None, None, None, 2]})

    result = merge_prices_with_ticker_df(prices, df)

    assert_series_equal(expected["date_day"], result["date_day"])
    assert_series_equal(expected["compound"], result["compound"])

    expected = pd.Series(["both"] + ["left_only"] * 7 + ["both"])
    assert_series_equal(result["_merge"], expected, check_dtype=False, check_names=False, check_categorical=False)


def test_remove_excluded_ticker():
    ticker_1 = Ticker("exclude", None)
    ticker_1.exclude = True

    ticker_2 = Ticker("do not exclude", None)
    ticker_2.exclude = False

    ticker = [ticker_1, ticker_2]

    result = remove_excluded_ticker.run(ticker)
    assert len(result) == 1
    assert result[0].name == "do not exclude"


def test_backfill_availability():
    df = pd.DataFrame({"available": [None, None, False, None, None, True, True]})
    result = backfill_availability.run(Ticker(None, df))
    expected = pd.Series([False, False, False, False, False, True, True])
    assert_series_equal(result.df["available"], expected, check_names=False)

    df = pd.DataFrame({"available": [False, None, False, None, None, True, True]})
    result = backfill_availability.run(Ticker(None, df))
    expected = pd.Series([False, False, False, False, False, True, True])
    assert_series_equal(result.df["available"], expected, check_names=False)

    df = pd.DataFrame({"available": [False, None, False, None, None, True, None, True]})
    result = backfill_availability.run(Ticker(None, df))
    expected = pd.Series([False, False, False, False, False, True, True, True])
    assert_series_equal(result.df["available"], expected, check_names=False)


def test_assign_price_col():
    df = pd.DataFrame({"test_1": [1, 2], "test_2": [3, 4]})
    result = assign_price_col.run(Ticker(None, df), price_col="test_1")
    expected = pd.DataFrame({"test_1": [1, 2], "test_2": [3, 4], "price": [1, 2]})

    assert_frame_equal(result.df, expected)


def test_mark_tradeable_days():
    df = pd.DataFrame([{date_day_col: Timestamp('2021-06-04 00:00:00')},
                       {date_day_col: Timestamp('2021-06-05 00:00:00')},
                       {date_day_col: Timestamp('2021-06-06 00:00:00')},
                       {date_day_col: Timestamp('2021-06-07 00:00:00')}])
    result = mark_tradeable_days.run(Ticker(None, df))
    expected = pd.Series([True, False, False, True])
    assert_series_equal(result.df["tradeable"], expected, check_names=False)

    assert set(result.df.columns) == {date_day_col, "tradeable"}

    df = pd.DataFrame([{date_day_col: Period('2021-01-28', 'D')},
                       {date_day_col: Period('2021-01-29', 'D')},
                       {date_day_col: Period('2021-02-01', 'D')},
                       {date_day_col: Period('2021-02-02', 'D')},
                       {date_day_col: Period('2021-02-03', 'D')},
                       {date_day_col: Period('2021-02-04', 'D')},
                       {date_day_col: Period('2021-02-05', 'D')},
                       {date_day_col: Period('2021-02-08', 'D')}])
    result = mark_tradeable_days.run(Ticker(None, df))
    expected = pd.Series([True] * 8)
    assert_series_equal(result.df["tradeable"], expected, check_names=False)

    assert set(result.df.columns) == {date_day_col, "tradeable"}


def test_forward_fill_price_data():
    df = pd.DataFrame([{date_col: Timestamp('2021-06-04 00:00:00'), "price": 5},
                       {date_col: Timestamp('2021-06-05 00:00:00'), "price": None},
                       {date_col: Timestamp('2021-06-06 00:00:00'), "price": None},
                       {date_col: Timestamp('2021-06-07 00:00:00'), "price": 10}])
    result = forward_fill_price_data.run(Ticker(None, df), price_data_cols=["price"])
    expected = df = pd.DataFrame([{date_col: Timestamp('2021-06-04 00:00:00'), "price": 5.0},
                                  {date_col: Timestamp('2021-06-05 00:00:00'), "price": 5.0},
                                  {date_col: Timestamp('2021-06-06 00:00:00'), "price": 5.0},
                                  {date_col: Timestamp('2021-06-07 00:00:00'), "price": 10.0}])
    assert_frame_equal(result.df, expected)


def test_mark_ticker_where_all_prices_are_nan():
    df = pd.DataFrame({"price": [None, None, None]})
    ticker = Ticker(None, df)
    result = mark_ticker_where_all_prices_are_nan.run(ticker)
    assert result.exclude is True

    df = pd.DataFrame({"price": [None, None, 1]})
    ticker = Ticker(None, df)
    result = mark_ticker_where_all_prices_are_nan.run(ticker)
    assert result.exclude is False


def test_mark_ipo_ticker():
    df = pd.DataFrame({"price": [None, None, None]})
    ticker = Ticker(None, df)
    result = mark_ipo_ticker.run(ticker)
    assert result.exclude is True

    df = pd.DataFrame({"price": [None, None, 1]})
    ticker = Ticker(None, df)
    result = mark_ipo_ticker.run(ticker)
    assert result.exclude is True

    df = pd.DataFrame({"price": [1, 1]})
    ticker = Ticker(None, df)
    result = mark_ipo_ticker.run(ticker)
    assert result.exclude is False


def test_drop_irrelevant_columns():
    # No test needed since it's basically a plain pandas function call
    pass


def test_fill_missing_sentiment_data():
    df = pd.DataFrame({"data": [None, 10, None]})
    result = fill_missing_sentiment_data.run(Ticker(None, df), ["data"])
    expected = pd.Series([0.0, 10.0, 0.0])

    assert_series_equal(result.df["data"], expected, check_names=False)


def test_assert_no_nan():
    df = pd.DataFrame({"data": [None, 10, None]})
    ticker = Ticker(None, df)
    with pytest.raises(AssertionError):
        assert_no_nan.run(ticker)

    df = pd.DataFrame({"data": [0, 0]})
    ticker = Ticker(None, df)


def test_add_metric_rel_price_change():
    df = pd.DataFrame({"price": [1, 2, 3, 4, 5, 10]})
    ticker = Ticker(None, df)
    result = add_metric_rel_price_change.run(ticker)
    expected = pd.Series([0.0, 1.0, 0.5, 0.333333, 0.25, 1.0])
    assert_series_equal(result.df["price_rel_change"], expected, check_exact=False, check_names=False)


def test_scale_price_data():
    df = pd.DataFrame({"price": [1, 2, 3, 4, 5, 10]})
    ticker = Ticker(None, df)

    # Without dropping
    result = scale_price_data.run(ticker, price_data_columns=["price"], drop_unscaled_cols=False)

    expected = pd.DataFrame({"price": [1, 2, 3, 4, 5, 10],
                             "price_scaled": [0.00000, 0.11111, 0.22222, 0.333333, 0.444444, 1.00000]})

    assert_frame_equal(result[0].df, expected)
    assert result[1] == ["price", "price_scaled"]

    # With dropping
    result = scale_price_data.run(ticker, price_data_columns=["price"], drop_unscaled_cols=True)

    expected = pd.DataFrame({"price_scaled": [0.00000, 0.11111, 0.22222, 0.333333, 0.444444, 1.00000]})

    assert_frame_equal(result[0].df, expected)
    assert result[1] == ["price_scaled"]


def test_make_sequence():
    # For more tests see test_sequences
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5], "available": [False, False, True, True, True],
                       "price": [10, 11, 12, 13, 14], "tradeable": [True] * 5})
    ticker = Ticker(None, df)

    result = make_sequences.run(ticker, 3, False, columns_to_be_excluded_from_sequences=["available", "tradeable"],
                                price_column="price")

    expected_flat_sequence = [
        pd.DataFrame(
            {"dummy/0": [1], "dummy/1": [2], "dummy/2": [3], "price/0": [10], "price/1": [11], "price/2": [12]}),
        pd.DataFrame(
            {"dummy/1": [2], "dummy/2": [3], "dummy/3": [4], "price/1": [11], "price/2": [12], "price/3": [13]}),
        pd.DataFrame(
            {"dummy/2": [3], "dummy/3": [4], "dummy/4": [5], "price/2": [12], "price/3": [13], "price/4": [14]}),
    ]

    for r, e in zip(result.sequences, expected_flat_sequence):
        r.flat.columns = e.columns  # r uses multi-level index, e doesn't (doesn't matter for the comparison tho)
        assert_frame_equal(r.flat, e, check_column_type=False)

    expected_arr_sequence = [
        pd.DataFrame({"dummy": [1, 2, 3], "price": [10, 11, 12]}),
        pd.DataFrame({"dummy": [2, 3, 4], "price": [11, 12, 13]}),
        pd.DataFrame({"dummy": [3, 4, 5], "price": [12, 13, 14]})
    ]

    for r, e in zip(result.sequences, expected_arr_sequence):
        assert_frame_equal(r.arr.reset_index(drop=True), e)


def test_make_sequences_which():
    df = pd.DataFrame({"dummy": [1, 2, 3, 4, 5], "available": [False, False, True, True, True],
                       "price": [10, 11, 12, 13, 14], "tradeable": [True] * 5})
    ticker = Ticker(None, df)

    result = make_sequences.run(ticker, 3, False, columns_to_be_excluded_from_sequences=["available", "tradeable"],
                                price_column="price", which="flat")

    for r in result.sequences:
        assert r.flat is not None
        assert r.arr is None

    result = make_sequences.run(ticker, 3, False, columns_to_be_excluded_from_sequences=["available", "tradeable"],
                                price_column="price", which="arr")

    for r in result.sequences:
        assert r.flat is None
        assert r.arr is not None

    result = make_sequences.run(ticker, 3, False, columns_to_be_excluded_from_sequences=["available", "tradeable"],
                                price_column="price", which="all")

    for r in result.sequences:
        assert r.flat is not None
        assert r.arr is not None


def test_remove_old_price_col_from_price_data_columns():
    price_data_columns = ["Close", "Open"]
    price_column = "Close"

    result = remove_old_price_col_from_price_data_columns.run(price_data_columns, price_column)
    assert result == ["Open", "price"]


def test_mark_sentiment_data_available_column():
    df = pd.DataFrame({"price": [1, 2, 3, 4, 5, 10], "dummy": [None, None, 0.14, 10.4, None, None]})
    ticker = Ticker(None, df)

    result = mark_sentiment_data_available_days.run(ticker, sentiment_data_columns=["dummy"])
    assert_series_equal(result.df["sentiment_data_available"], pd.Series([False, False, True, True, False, False]),
                        check_names=False, check_dtype=False)
