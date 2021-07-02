import pandas as pd

from pandas import Period, Timestamp
import datetime
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
from preprocessing.price_data.cached_stock_data import CachedStockData


def test_generate_date_range():
    start = Period('2021-05-10', 'D')
    end = Period('2021-05-15', 'D')
    csd = CachedStockData(start_date=start, end_date=end, ticker=None, live=None)
    result = csd.generate_date_range()

    expected = pd.date_range(start.to_timestamp(), end.to_timestamp(), freq="D")

    assert (result == expected).all()

    csd = CachedStockData(start_date=start, end_date=end, ticker=None, live=None)
    result = csd.generate_date_range()

    expected = pd.date_range(start.to_timestamp(), Period('2021-05-14', 'D').to_timestamp(), freq="D")

    with pytest.raises(ValueError):
        assert (result == expected).all()


def test_get_missing_dates():
    start = Period('2021-05-10', 'D')
    end = Period('2021-05-15', 'D')

    df = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'), Period('2021-05-11', 'D'), Period('2021-05-12', 'D'),
                                    Period('2021-05-15', 'D')]})
    rng = pd.date_range(start.to_timestamp(), end.to_timestamp(), freq="D")
    csd = CachedStockData(None, None, None, None)
    result = csd.get_missing_dates(df, rng)

    expected = pd.DataFrame({"date_day": [Period('2021-05-13', 'D'), Period('2021-05-14', 'D')]})

    assert_frame_equal(expected, result)


def test_filter_weekends():
    df = pd.DataFrame({"date_day": [Period('2021-07-01', 'D'), Period('2021-07-02', 'D'), Period('2021-07-03', 'D'),
                                    Period('2021-07-04', 'D'), Period('2021-07-05', 'D')]})
    csd = CachedStockData(None, None, None, None)
    result = csd.filter_weekends(df)
    expected = pd.DataFrame({"date_day": [Period('2021-07-01', 'D'), Period('2021-07-02', 'D'),
                                          Period('2021-07-05', 'D')]})

    assert_frame_equal(expected, result.reset_index(drop=True))


def test_get_sequences():
    df = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'), Period('2021-05-11', 'D'), Period('2021-05-12', 'D'),
                                    Period('2021-05-15', 'D'), Period('2021-05-16', 'D'), Period('2021-05-17', 'D'),
                                    Period('2021-05-20', 'D')]})
    csd = CachedStockData(None, None, None, None)
    result = csd.get_consecutive_sequences(df)

    expected = [[Timestamp('2021-05-10 00:00:00'), Timestamp('2021-05-11 00:00:00'), Timestamp('2021-05-12 00:00:00')],
                [Timestamp('2021-05-15 00:00:00'), Timestamp('2021-05-16 00:00:00'), Timestamp('2021-05-17 00:00:00')],
                [Timestamp('2021-05-20 00:00:00')]]

    assert result == expected

    # Check ordering
    df = pd.DataFrame({"date_day": [Period('2021-05-11', 'D'), Period('2021-05-10', 'D'), Period('2021-05-12', 'D'),
                                    Period('2021-05-16', 'D'), Period('2021-05-15', 'D'), Period('2021-05-17', 'D'),
                                    Period('2021-05-20', 'D')]})
    csd = CachedStockData(None, None, None, None)
    result = csd.get_consecutive_sequences(df)

    expected = [[Timestamp('2021-05-10 00:00:00'), Timestamp('2021-05-11 00:00:00'), Timestamp('2021-05-12 00:00:00')],
                [Timestamp('2021-05-15 00:00:00'), Timestamp('2021-05-16 00:00:00'), Timestamp('2021-05-17 00:00:00')],
                [Timestamp('2021-05-20 00:00:00')]]

    assert result == expected

    # Check one entry
    df = pd.DataFrame({"date_day": [Period('2021-05-11', 'D')]})
    csd = CachedStockData(None, None, None, None)
    result = csd.get_consecutive_sequences(df)

    expected = [[Timestamp('2021-05-11 00:00:00')]]

    assert result == expected


def test_get_from_cache():
    start = Period('2021-05-11', 'D')
    end = Period('2021-05-12', 'D')
    csd = CachedStockData(start_date=start, end_date=end,
                          ticker="AAPL", live=False)
    csd.initialize_cache(":memory:")

    df = pd.DataFrame({'ticker': ["AAPL", "AAPL", "AAPL", "AAPL"],
                       "Close": [1, 2, 3, 4],
                       "date_day": [Period('2021-05-10', 'D'),
                                    Period('2021-05-11', 'D'),
                                    Period('2021-05-12', 'D'),
                                    Period('2021-05-14', 'D')]})

    csd.c.append(df)

    result = csd.get_from_cache()
    expected = pd.DataFrame({"date_day": [Period('2021-05-11', 'D'),
                                          Period('2021-05-12', 'D')]})
    result = result.reset_index(drop=True)
    assert_series_equal(result["date_day"], expected["date_day"])


def test_download_one_sequence():
    csd = CachedStockData(ticker="AAPL", start_date=None, end_date=None, live=None)
    csd.initialize_cache(":memory:")

    sequences = sequences = [[Timestamp('2021-05-11 00:00:00')]]
    csd.download(sequences)
    result = csd.c.get("AAPL")
    expected = pd.DataFrame({'Open': [129.41000366210938, 123.5],
                             'High': [129.5399932861328, 126.2699966430664],
                             'Low': [126.80999755859375, 122.7699966430664],
                             'Close': [126.8499984741211, 125.91000366210938],
                             'Adj Close': [126.8499984741211, 125.91000366210938],
                             'Volume': [88071200, 126142800],
                             'date_day': [Period('2021-05-10', 'D'), Period('2021-05-11', 'D')],
                             'ticker': ['AAPL', 'AAPL']})

    assert_frame_equal(result, expected)


def test_download():
    csd = CachedStockData(ticker="AAPL", start_date=None, end_date=None, live=None)
    csd.initialize_cache(":memory:")

    sequences = [[Timestamp('2021-05-11 00:00:00'), Timestamp('2021-05-12 00:00:00')]]
    csd.download(sequences)
    result = csd.c.get("AAPL")
    expected = pd.DataFrame({'Open': [129.41000366210938, 123.5, 123.4000015258789],
                             'High': [129.5399932861328, 126.2699966430664, 124.63999938964844],
                             'Low': [126.80999755859375, 122.7699966430664, 122.25],
                             'Close': [126.8499984741211, 125.91000366210938, 122.7699966430664],
                             'Adj Close': [126.8499984741211, 125.91000366210938, 122.7699966430664],
                             'Volume': [88071200, 126142800, 112172300],
                             'date_day': [Period('2021-05-10', 'D'), Period('2021-05-11', 'D'),
                                          Period('2021-05-12', 'D')], 'ticker': ['AAPL', 'AAPL', 'AAPL']})
    assert_frame_equal(result, expected)


def test_get():
    csd = CachedStockData(ticker="AAPL", start_date=Period('2021-05-10', 'D'), end_date=Period('2021-05-15', 'D'),
                          live=False)
    csd.initialize_cache(":memory:")

    df = pd.DataFrame({'ticker': ["AAPL", "AAPL"],
                       "Adj Close": [1, 2],
                       "Open": [1, 2],
                       "Volume": [1, 2],
                       "Close": [1, 2],
                       "High": [1, 2],
                       "Low": [1, 2],
                       "date_day": [Period('2021-05-10', 'D'),
                                    Period('2021-05-11', 'D')]})
    csd.c.append(df)

    result = csd.get()
    expected = pd.DataFrame(
        {'ticker': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL'],
         'Adj Close': [1.0, 2.0, 122.7699966430664, 124.97000122070312, 127.44999694824219],
         'Open': [1.0, 2.0, 123.4000015258789, 124.58000183105469, 126.25],
         'Volume': [1, 2, 112172300, 105861300, 81918000],
         'Close': [1.0, 2.0, 122.7699966430664, 124.97000122070312, 127.44999694824219],
         'High': [1.0, 2.0, 124.63999938964844, 126.1500015258789, 127.88999938964844],
         'Low': [1.0, 2.0, 122.25, 124.26000213623047, 125.8499984741211],
         'date_day': [Period('2021-05-10', 'D'), Period('2021-05-11', 'D'), Period('2021-05-12', 'D'),
                      Period('2021-05-13', 'D'), Period('2021-05-14', 'D')]}
    )

    # 2021-05-15 is weekend

    assert_frame_equal(result, expected)
