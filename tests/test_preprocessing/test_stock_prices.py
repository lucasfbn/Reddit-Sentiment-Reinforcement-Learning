import datetime

import pandas as pd
import pytest
from pandas import Period
from pandas.testing import assert_series_equal

from preprocessing.stock_prices import StockPrices, OldDataException, MissingDataException


def test_historic():
    df = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'),
                                    Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="GME", start_date=df["date_day"].min(),
                     end_date=df["date_day"].max() + datetime.timedelta(days=1), live=False)
    prices = sp.download()

    assert prices.loc[0, "date_day"] == Period('2021-05-10', 'D')
    assert prices.loc[len(prices) - 1, "date_day"] == Period('2021-05-20', 'D')


def test_offset():
    df = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'),
                                    Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="GME", start_date=df["date_day"].min() - datetime.timedelta(days=10),
                     end_date=df["date_day"].max() + datetime.timedelta(days=1), live=False)
    prices = sp.download()

    assert prices.loc[0, "date_day"] == Period('2021-04-29', 'D')
    assert prices.loc[len(prices) - 1, "date_day"] == Period('2021-05-20', 'D')


def test_historic_last_date_weekend():
    # '2021-06-05' is on a weekend -> no price data
    df = pd.DataFrame({"date_day": [Period('2021-06-01', 'D'),
                                    Period('2021-06-05', 'D')]})

    sp = StockPrices(ticker_name="GME", start_date=df["date_day"].min(),
                     end_date=df["date_day"].max() + datetime.timedelta(days=1), live=False)
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

    sp = StockPrices(ticker_name="GME", start_date=df["date_day"].min(),
                     end_date=df["date_day"].max() + datetime.timedelta(days=1), live=True)

    with pytest.raises(OldDataException):
        prices = sp.download()


def test_historic_missing_data():
    df = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'),
                                    Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="AHJZT", start_date=df["date_day"].min(),
                     end_date=df["date_day"].max() + datetime.timedelta(days=1), live=False)

    with pytest.raises(MissingDataException):
        prices = sp.download()


def test_live_missing_data():
    df = pd.DataFrame({"date_day": [Period('2021-05-10', 'D'),
                                    Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="AHJZT", start_date=df["date_day"].min(),
                     end_date=df["date_day"].max() + datetime.timedelta(days=1), live=True)

    with pytest.raises(MissingDataException):
        prices = sp.download()
