import datetime

import pandas as pd
import pytest
from pandas import Period
from pandas.testing import assert_series_equal

from preprocessing.stock_prices import StockPrices, OldDataException, MissingDataException


def test_historic():
    df = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'),
                                    Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="GME", ticker_df=df, start_offset=0, live=False)
    prices = sp.download()

    assert prices.loc[0, "date_day"] == Period('2021-05-10', 'D')
    assert prices.loc[len(prices) - 1, "date_day"] == Period('2021-05-20', 'D')


def test_offset():
    df = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'),
                                    Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="GME", ticker_df=df, start_offset=10, live=False)
    prices = sp.download()

    assert prices.loc[0, "date_day"] == Period('2021-04-19', 'D')
    assert prices.loc[len(prices) - 1, "date_day"] == Period('2021-05-20', 'D')


def test_historic_last_date_weekend():
    # '2021-06-05' is on a weekend -> no price data
    df = pd.DataFrame({"date_day": [Period('2021-06-01', 'D'),
                                    Period('2021-06-05', 'D')]})

    sp = StockPrices(ticker_name="GME", ticker_df=df, start_offset=0, live=False)
    prices = sp.download()

    assert prices.loc[0, "date_day"] == Period('2021-06-01', 'D')
    assert prices.loc[len(prices) - 1, "date_day"] == Period('2021-06-04', 'D')


def test_live_too_early():
    """
    ONLY WORKS PRIOR TO (AMERICAN) MARKET OPENINGS.
    """

    if datetime.datetime.now().hour >= 15:
        return

    df = pd.DataFrame({"date_day": [Period('2021-06-05', 'D'),
                                    Period('2021-06-06', 'D')]})

    sp = StockPrices(ticker_name="GME", ticker_df=df, start_offset=0, live=True)

    with pytest.raises(OldDataException):
        prices = sp.download()


def test_historic_missing_data():
    df = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'),
                                    Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="AHJZT", ticker_df=df, start_offset=0, live=False)

    with pytest.raises(MissingDataException):
        prices = sp.download()


def test_live_missing_data():
    df = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'),
                                    Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="AHJZT", ticker_df=df, start_offset=0, live=True)

    with pytest.raises(MissingDataException):
        prices = sp.download()


def test_merge():
    df = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'),
                                    Period('2021-05-20', 'D')],
                       "compound": [1, 2]})

    sp = StockPrices(ticker_name="GME", ticker_df=df, start_offset=0, live=False)
    prices = sp.download()
    merged = sp.merge()

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

    assert_series_equal(expected["date_day"], merged["date_day"])
    assert_series_equal(expected["compound"], merged["compound"])


def test_merge_indicator():
    df = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'),
                                    Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="GME", ticker_df=df, start_offset=0, live=False)
    sp.download()

    result = sp.merge()
    expected = pd.Series(["both"] + ["left_only"] * 7 + ["both"])
    assert_series_equal(result["_merge"], expected, check_dtype=False, check_names=False, check_categorical=False)
