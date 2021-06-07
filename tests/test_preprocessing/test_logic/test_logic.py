import pandas as pd
from preprocessing.logic.logic import *
from pandas import Timestamp, Period
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest


def test_add_time():
    df = pd.DataFrame({"end": ["2021-05-20 05:59"]})
    result = add_time.run(df)
    expected = pd.DataFrame([{'end': '2021-05-20 05:59', 'date': Timestamp('2021-05-20 05:59:00'),
                              "date_day": Period('2021-05-20', 'D')}])
    assert_frame_equal(result, expected)


def test_shift_time():
    df = pd.DataFrame([{'date': Timestamp('2021-05-20 08:00:00')},
                       {'date': Timestamp('2021-05-20 12:00:00')},
                       {'date': Timestamp('2021-05-21 08:00:00')}])
    result = shift_time.run(df, start_hour=8, start_min=0)
    expected = pd.DataFrame([{'date': Timestamp('2021-05-20 08:00:00'),
                              'date_shifted': Timestamp('2021-05-21 00:00:00'),
                              'date_shifted_day': Period('2021-05-21', 'D')},
                             {'date': Timestamp('2021-05-20 12:00:00'),
                              'date_shifted': Timestamp('2021-05-21 04:00:00'),
                              'date_shifted_day': Period('2021-05-21', 'D')},
                             {'date': Timestamp('2021-05-21 08:00:00'),
                              'date_shifted': Timestamp('2021-05-22 00:00:00'),
                              'date_shifted_day': Period('2021-05-22', 'D')}])
    assert_frame_equal(result, expected)


def test_get_min_max_time():
    df = pd.DataFrame([{'date_shifted_day': Period('2021-05-21', 'D')},
                       {'date_shifted_day': Period('2021-05-19', 'D')},
                       {'date_shifted_day': Period('2021-05-22', 'D')}])
    result_0, result_1 = get_min_max_time.run(df)
    assert Period('2021-05-19', 'D') == result_0 and Period('2021-05-22', 'D') == result_1


def test_scale_daywise():
    df = pd.DataFrame([
        {'date_shifted_day': Period('2021-05-21', 'D'), 'to_be_scaled': 0, "exclude": 999},
        {'date_shifted_day': Period('2021-05-21', 'D'), 'to_be_scaled': 1, "exclude": 777},
        {'date_shifted_day': Period('2021-05-21', 'D'), 'to_be_scaled': 2, "exclude": 888}
    ])

    # Scale without dropping cols afterwards
    result = scale_daywise.run(df, excluded_cols_from_scaling=["date_shifted_day", "exclude"], drop_scaled_cols=False)
    expected = pd.DataFrame([
        {'date_shifted_day': Period('2021-05-21', 'D'), 'to_be_scaled': 0, "exclude": 999,
         "to_be_scaled_scaled": 0.00000},
        {'date_shifted_day': Period('2021-05-21', 'D'), 'to_be_scaled': 1, "exclude": 777,
         "to_be_scaled_scaled": 0.50000},
        {'date_shifted_day': Period('2021-05-21', 'D'), 'to_be_scaled': 2, "exclude": 888,
         "to_be_scaled_scaled": 1.00000}
    ])
    assert_frame_equal(result, expected)

    # Scale with dropping cols afterwards
    result = scale_daywise.run(df, excluded_cols_from_scaling=["date_shifted_day", "exclude"], drop_scaled_cols=True)
    expected = pd.DataFrame([
        {'date_shifted_day': Period('2021-05-21', 'D'), "exclude": 999,
         "to_be_scaled_scaled": 0.00000},
        {'date_shifted_day': Period('2021-05-21', 'D'), "exclude": 777,
         "to_be_scaled_scaled": 0.50000},
        {'date_shifted_day': Period('2021-05-21', 'D'), "exclude": 888,
         "to_be_scaled_scaled": 1.00000}
    ])
    assert_frame_equal(result, expected)


def test_grp_by_ticker():
    df = pd.DataFrame([{'ticker': "GME", 'date_shifted': Timestamp('2021-05-21 00:00:00'), "dummy_value": 1},
                       {'ticker': "GME", 'date_shifted': Timestamp('2021-05-20 00:00:00'), "dummy_value": 1},
                       {'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-19 00:00:00'), "dummy_value": 1},
                       {'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-21 00:00:00'), "dummy_value": 1},
                       {'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-20 00:00:00'), "dummy_value": 1}])
    result = grp_by_ticker.run(df)

    expected_df_1 = pd.DataFrame([{'ticker': "GME", 'date_shifted': Timestamp('2021-05-20 00:00:00'), "dummy_value": 1},
                                  {'ticker': "GME", 'date_shifted': Timestamp('2021-05-21 00:00:00'),
                                   "dummy_value": 1}])

    expected_df_2 = pd.DataFrame(
        [{'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-19 00:00:00'), "dummy_value": 1},
         {'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-20 00:00:00'), "dummy_value": 1},
         {'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-21 00:00:00'), "dummy_value": 1}])

    assert result[0].name == "GME" and result[1].name == "TSLA"
    assert_frame_equal(result[0].df.reset_index(drop=True), expected_df_1)
    assert_frame_equal(result[1].df.reset_index(drop=True), expected_df_2)


def test_drop_ticker_with_too_few_data():
    df_1 = pd.DataFrame([{'ticker': "GME", 'date_shifted': Timestamp('2021-05-20 00:00:00'), "dummy_value": 1},
                         {'ticker': "GME", 'date_shifted': Timestamp('2021-05-21 00:00:00'), "dummy_value": 1}])
    ticker_1 = Ticker(name="GME", df=df_1)
    df_2 = pd.DataFrame(
        [{'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-19 00:00:00'), "dummy_value": 1},
         {'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-20 00:00:00'), "dummy_value": 1},
         {'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-21 00:00:00'), "dummy_value": 1}])
    ticker_2 = Ticker(name="TSLA", df=df_2)

    result = drop_ticker_with_too_few_data.run([ticker_1, ticker_2], ticker_min_len=3)
    assert len(result) == 1 and result[0].name == "TSLA"


def test_mark_trainable_days():
    df = pd.DataFrame([{'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-18 00:00:00')},
                       {'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-19 00:00:00')},
                       {'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-20 00:00:00')},
                       {'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-21 00:00:00')}])
    result = mark_trainable_days.run(Ticker(name="TSLA", df=df), ticker_min_len=2)
    expected = pd.Series([False, False, True, True], name="available")
    assert_series_equal(result.df["available"], expected)

    # Test not sorted by date
    df = pd.DataFrame([{'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-21 00:00:00')},
                       {'ticker': "TSLA", 'date_shifted': Timestamp('2021-05-19 00:00:00')}])

    with pytest.raises(AssertionError):
        mark_trainable_days.run(Ticker(name="TSLA", df=df), ticker_min_len=1)
