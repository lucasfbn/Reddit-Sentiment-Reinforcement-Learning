import pandas as pd
from pandas import Period
from pandas.testing import assert_frame_equal

from preprocessing.price_data.cache import Cache


def test_first_append():
    c = Cache(db_path=":memory:")
    df = pd.DataFrame(
        {'ticker': ["AAPL", "TSLA", "GOE"], "Close": [1, 2, 3], "date_day_shifted": [Period('2021-05-10', 'D'),
                                                                                     Period('2021-05-11', 'D'),
                                                                                     Period('2021-05-12', 'D')]})
    c.append(df)
    result = c.get_all()
    print(result.dtypes)
    print(df.dtypes)

    assert_frame_equal(df, result)


def test_several_appends():
    c = Cache(db_path=":memory:")
    df = pd.DataFrame(
        {'ticker': ["AAPL", "TSLA", "GOE"], "Close": [1, 2, 3], "date_day_shifted": [Period('2021-05-10', 'D'),
                                                                                     Period('2021-05-11', 'D'),
                                                                                     Period('2021-05-12', 'D')]})
    c.append(df, drop_duplicates=False)
    c.append(df, drop_duplicates=False)
    c.append(df, drop_duplicates=False)
    result = c.get_all()

    expected = pd.DataFrame(
        {'ticker': ["AAPL", "TSLA", "GOE"] * 3,
         "Close": [1, 2, 3] * 3, "date_day_shifted": [Period('2021-05-10', 'D'),
                                                      Period('2021-05-11', 'D'),
                                                      Period('2021-05-12', 'D')] * 3})

    assert_frame_equal(expected, result)

    c = Cache(db_path=":memory:")
    df = pd.DataFrame(
        {'ticker': ["AAPL", "TSLA", "GOE"], "Close": [1, 2, 3], "date_day_shifted": [Period('2021-05-10', 'D'),
                                                                                     Period('2021-05-11', 'D'),
                                                                                     Period('2021-05-12', 'D')]})
    c.append(df)
    c.append(df)
    c.append(df)
    result = c.get_all()

    expected = pd.DataFrame(
        {'ticker': ["AAPL", "TSLA", "GOE"],
         "Close": [1, 2, 3], "date_day_shifted": [Period('2021-05-10', 'D'),
                                                  Period('2021-05-11', 'D'),
                                                  Period('2021-05-12', 'D')]})

    assert_frame_equal(expected, result)


def test_drop_duplicates():
    c = Cache(db_path=":memory:")
    df = pd.DataFrame({"ticker": ["AAPL"] * 3, "Close": [1, 2, 3], "date_day_shifted": [Period('2021-05-10', 'D'),
                                                                                        Period('2021-05-11', 'D'),
                                                                                        Period('2021-05-12', 'D')]})
    c.append(df, drop_duplicates=False)
    c.append(df, drop_duplicates=False)
    c.append(df, drop_duplicates=False)
    result = c.get_all()

    expected = pd.DataFrame(
        {"ticker": ["AAPL"] * 9, "Close": [1, 2, 3] * 3, "date_day_shifted": [Period('2021-05-10', 'D'),
                                                                              Period('2021-05-11', 'D'),
                                                                              Period('2021-05-12', 'D')] * 3})

    assert_frame_equal(expected, result)

    c.drop_duplicates()
    result = c.get_all()
    assert_frame_equal(df, result)

    def test_get_specific():
        c = Cache(db_path=":memory:")
        df = pd.DataFrame(
            {'ticker': ["AAPL", "TSLA", "GOE"], "Close": [1, 2, 3], "date_day_shifted": [Period('2021-05-10', 'D'),
                                                                                         Period('2021-05-11', 'D'),
                                                                                         Period('2021-05-12', 'D')]})
        c.append(df)
        result = c.get("TSLA")

        expected = pd.DataFrame({"Close": [2], "date_day_shifted": [Period('2021-05-11', 'D')]})

        assert_frame_equal(expected, result)


def test_drop_tail():
    c = Cache(":memory:")

    df = pd.DataFrame({'ticker': ["AAPL", "AAPL", "AAPL", "TSLA", "TSLA", "TSLA"],
                       "Adj_Close": [1, 2, 3, 4, 5, 6],
                       "date_day_shifted": [Period('2021-05-11', 'D'), Period('2021-05-10', 'D'),
                                            Period('2021-05-09', 'D'), Period('2021-05-11', 'D'),
                                            Period('2021-05-10', 'D'), Period('2021-05-09', 'D')]})

    c.append(df)
    c.drop_tail(2)

    expected = pd.DataFrame({"ticker": ["AAPL", "TSLA"],
                             "Adj_Close": [3, 6],
                             "date_day_shifted": [Period('2021-05-09', 'D'), Period('2021-05-09', 'D')]})

    assert_frame_equal(expected, c.get_all())
