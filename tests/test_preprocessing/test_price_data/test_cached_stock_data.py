import pandas as pd
from pandas import Period
from pandas.testing import assert_frame_equal

from preprocessing.price_data.cached_stock_data import CachedStockData


def test_get():
    csd = CachedStockData(ticker="AAPL", start_date=Period('2021-05-10', 'D'), end_date=Period('2021-05-15', 'D'),
                          live=False)
    csd.initialize_cache(":memory:")

    df = pd.DataFrame({'ticker': ["AAPL", "AAPL"],
                       "Adj_Close": [1, 2],
                       "Open": [1, 2],
                       "Volume": [1, 2],
                       "Close": [1, 2],
                       "High": [1, 2],
                       "Low": [1, 2],
                       "date_day_shifted": [Period('2021-05-10', 'D'),
                                            Period('2021-05-11', 'D')]})
    csd.c.append(df)
    result = csd.get()

    expected = pd.DataFrame(
        {'Adj_Close': [1.0, 125.53842163085938, 122.40768432617188, 124.6011962890625, 127.07386779785156],
         'Open': [1.0, 123.5, 123.4000015258789, 124.58000183105469, 126.25],
         'Volume': [1, 126142800, 112172300, 105861300, 81918000],
         'Close': [1.0, 125.91000366210938, 122.7699966430664, 124.97000122070312, 127.44999694824219],
         'High': [1.0, 126.2699966430664, 124.63999938964844, 126.1500015258789, 127.88999938964844],
         'Low': [1.0, 122.7699966430664, 122.25, 124.26000213623047, 125.8499984741211],
         'date_day_shifted': [Period('2021-05-10', 'D'), Period('2021-05-11', 'D'), Period('2021-05-12', 'D'),
                              Period('2021-05-13', 'D'), Period('2021-05-14', 'D')]}

    )

    # 2021-05-15 is weekend

    assert_frame_equal(result, expected)


def test_get_filter():
    csd = CachedStockData(ticker="AAPL", start_date=Period('2021-05-10', 'D'), end_date=Period('2021-05-17', 'D'),
                          live=False)
    csd.initialize_cache(":memory:")

    # Need some dummy data such that the table has columns
    df = pd.DataFrame({'ticker': ["AAPL"], "Adj_Close": [1], "Open": [1], "Volume": [1], "Close": [1],
                       "High": [1], "Low": [1], "date_day_shifted": [Period('2021-05-05', 'D')]})
    csd.c.append(df)

    result = csd.get()

    assert result["date_day_shifted"].loc[0] == Period('2021-05-10', 'D')
    assert result["date_day_shifted"].loc[len(result) - 1] == Period('2021-05-17', 'D')


def test_get_filling():
    """
    Tests whether the whole data gets filled from the last data on even though a later date was requested.
    """
    csd = CachedStockData(ticker="AAPL", start_date=Period('2021-05-10', 'D'), end_date=Period('2021-05-17', 'D'),
                          live=False)
    csd.initialize_cache(":memory:")

    # Need some dummy data such that the table has columns
    df = pd.DataFrame({'ticker': ["AAPL"], "Adj_Close": [1], "Open": [1], "Volume": [1], "Close": [1],
                       "High": [1], "Low": [1], "date_day_shifted": [Period('2021-05-05', 'D')]})
    csd.c.append(df)

    csd.get()

    result = csd.c.get("AAPL")

    expected = pd.DataFrame({'Adj_Close': [1.0, 129.52000427246094, 130.2100067138672, 126.8499984741211,
                                           125.91000366210938, 122.7699966430664, 124.97000122070312,
                                           127.44999694824219, 126.2699966430664],
                             'Open': [1.0, 127.88999938964844, 130.85000610351562, 129.41000366210938, 123.5,
                                      123.4000015258789, 124.58000183105469, 126.25, 126.81999969482422],
                             'Volume': [1, 78128300, 78973300, 88071200, 126142800, 112172300, 105861300, 81918000,
                                        74244600],
                             'Close': [1.0, 129.74000549316406, 130.2100067138672, 126.8499984741211,
                                       125.91000366210938, 122.7699966430664, 124.97000122070312, 127.44999694824219,
                                       126.2699966430664],
                             'High': [1.0, 129.75, 131.25999450683594, 129.5399932861328, 126.2699966430664,
                                      124.63999938964844, 126.1500015258789, 127.88999938964844, 126.93000030517578],
                             'Low': [1.0, 127.12999725341797, 129.47999572753906, 126.80999755859375, 122.7699966430664,
                                     122.25, 124.26000213623047, 125.8499984741211, 125.16999816894531],
                             'date_day_shifted': [Period('2021-05-05', 'D'), Period('2021-05-06', 'D'),
                                                  Period('2021-05-07', 'D'), Period('2021-05-10', 'D'),
                                                  Period('2021-05-11', 'D'), Period('2021-05-12', 'D'),
                                                  Period('2021-05-13', 'D'), Period('2021-05-14', 'D'),
                                                  Period('2021-05-17', 'D')]}
                            )

    assert_frame_equal(result, expected)


def test_edges():
    csd = CachedStockData(ticker="AAPL", start_date=Period('2021-05-10', 'D'), end_date=Period('2021-05-10', 'D'),
                          live=False)
    csd.initialize_cache(":memory:")

    # Need some dummy data such that the table has columns
    df = pd.DataFrame({'ticker': ["AAPL"], "Adj_Close": [1], "Open": [1], "Volume": [1], "Close": [1],
                       "High": [1], "Low": [1], "date_day_shifted": [Period('2021-05-05', 'D')]})
    csd.c.append(df)

    result = csd.get()

    assert len(result) == 1
    assert result["date_day_shifted"].loc[0] == Period('2021-05-10', 'D')


def test_non_existing_ticker():
    csd = CachedStockData(ticker="TSLA", start_date=Period('2021-02-03', 'D'), end_date=Period('2021-02-05', 'D'),
                          live=False)
    csd.initialize_cache(":memory:")

    # Need some dummy data such that the table has columns
    df = pd.DataFrame({'ticker': ["AAPL"], "Adj_Close": [1], "Open": [1], "Volume": [1], "Close": [1],
                       "High": [1], "Low": [1], "date_day_shifted": [Period('2021-05-05', 'D')]})
    csd.c.append(df)

    result = csd.get()

    assert result["date_day_shifted"].loc[0] == Period('2021-02-03', 'D')
    assert result["date_day_shifted"].loc[len(result) - 1] == Period('2021-02-05', 'D')

    # Check that the data got downloaded from the standard_date on
    result = csd.c.get("TSLA")
    assert result["date_day_shifted"].loc[0] == Period('2021-02-01', 'D')
    assert result["date_day_shifted"].loc[len(result) - 1] == Period('2021-02-05', 'D')


def test_live():
    csd = CachedStockData(ticker="AAPL", start_date=Period('2021-05-10', 'D'), end_date=Period('2021-05-17', 'D'),
                          live=True)
    csd.initialize_cache(":memory:")

    # Need some dummy data such that the table has columns
    df = pd.DataFrame({'ticker': ["AAPL"], "Adj_Close": [1], "Open": [1], "Volume": [1], "Close": [1],
                       "High": [1], "Low": [1], "date_day_shifted": [Period('2021-05-19', 'D')]})
    csd.c.append(df)

    result = csd.get()

    assert result["date_day_shifted"].loc[len(result) - 1] == Period.now("D")
