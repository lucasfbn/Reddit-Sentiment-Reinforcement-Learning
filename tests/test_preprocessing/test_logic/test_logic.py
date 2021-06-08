import pandas as pd
from preprocessing.logic.logic import *
from pandas import Timestamp, Period
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest


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


def test_get_min_max_time():
    df = pd.DataFrame([{date_day_shifted_col: Period('2021-05-21', 'D')},
                       {date_day_shifted_col: Period('2021-05-19', 'D')},
                       {date_day_shifted_col: Period('2021-05-22', 'D')}])
    result_0, result_1 = get_min_max_time.run(df)
    assert Period('2021-05-19', 'D') == result_0 and Period('2021-05-22', 'D') == result_1


def test_scale_daywise():
    df = pd.DataFrame([
        {date_day_shifted_col: Period('2021-05-21', 'D'), 'to_be_scaled': 0, "exclude": 999},
        {date_day_shifted_col: Period('2021-05-21', 'D'), 'to_be_scaled': 1, "exclude": 777},
        {date_day_shifted_col: Period('2021-05-21', 'D'), 'to_be_scaled': 2, "exclude": 888}
    ])

    # Scale without dropping cols afterwards
    result = scale_daywise.run(df, cols_to_be_scaled=["to_be_scaled"], drop_scaled_cols=False)
    expected = pd.DataFrame([
        {date_day_shifted_col: Period('2021-05-21', 'D'), 'to_be_scaled': 0, "exclude": 999,
         "to_be_scaled_scaled": 0.00000},
        {date_day_shifted_col: Period('2021-05-21', 'D'), 'to_be_scaled': 1, "exclude": 777,
         "to_be_scaled_scaled": 0.50000},
        {date_day_shifted_col: Period('2021-05-21', 'D'), 'to_be_scaled': 2, "exclude": 888,
         "to_be_scaled_scaled": 1.00000}
    ])
    assert_frame_equal(result, expected)

    # Scale with dropping cols afterwards
    result = scale_daywise.run(df, cols_to_be_scaled=["to_be_scaled"], drop_scaled_cols=True)
    expected = pd.DataFrame([
        {date_day_shifted_col: Period('2021-05-21', 'D'), "exclude": 999,
         "to_be_scaled_scaled": 0.00000},
        {date_day_shifted_col: Period('2021-05-21', 'D'), "exclude": 777,
         "to_be_scaled_scaled": 0.50000},
        {date_day_shifted_col: Period('2021-05-21', 'D'), "exclude": 888,
         "to_be_scaled_scaled": 1.00000}
    ])
    assert_frame_equal(result, expected)


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
    result = sort_ticker_df_chronologically.run(Ticker(None, df))
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
