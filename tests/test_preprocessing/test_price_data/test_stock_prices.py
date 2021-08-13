import datetime

import pandas as pd
import pytest
from pandas import Period

from preprocessing.price_data.stock_prices import StockPrices, OldDataException, MissingDataException


def test_historic():
    df = pd.DataFrame({"date_day_shifted": [Period('2021-05-10', 'D'),
                                            Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="GME", start_date=df["date_day_shifted"].min(),
                     end_date=df["date_day_shifted"].max(), live=False)
    prices = sp.download()

    assert prices.loc[0, "date_day_shifted"] == Period('2021-05-10', 'D')
    assert prices.loc[len(prices) - 1, "date_day_shifted"] == Period('2021-05-20', 'D')


def test_offset():
    df = pd.DataFrame({"date_day_shifted": [Period('2021-05-10', 'D'),
                                            Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="GME", start_date=df["date_day_shifted"].min() - datetime.timedelta(days=10),
                     end_date=df["date_day_shifted"].max(), live=False)
    prices = sp.download()

    assert prices.loc[0, "date_day_shifted"] == Period('2021-04-30', 'D')
    assert prices.loc[len(prices) - 1, "date_day_shifted"] == Period('2021-05-20', 'D')


def test_historic_last_date_weekend():
    # '2021-06-05' is on a weekend -> no price data
    df = pd.DataFrame({"date_day_shifted": [Period('2021-06-01', 'D'),
                                            Period('2021-06-05', 'D')]})

    sp = StockPrices(ticker_name="GME", start_date=df["date_day_shifted"].min(),
                     end_date=df["date_day_shifted"].max(), live=False)
    prices = sp.download()

    assert prices.loc[0, "date_day_shifted"] == Period('2021-06-01', 'D')
    assert prices.loc[len(prices) - 1, "date_day_shifted"] == Period('2021-06-04', 'D')


def test_live_too_early():
    """
    ONLY WORKS PRIOR TO (AMERICAN) MARKET OPENINGS.
    """

    if datetime.datetime.now().hour >= 15:
        return

    df = pd.DataFrame({"date_day_shifted": [Period('2021-06-05', 'D'),
                                            Period('2021-06-06', 'D')]})

    sp = StockPrices(ticker_name="GME", start_date=df["date_day_shifted"].min(),
                     end_date=df["date_day_shifted"].max(), live=True)

    with pytest.raises(OldDataException):
        prices = sp.download()


def test_historic_missing_data():
    df = pd.DataFrame({"date_day_shifted": [Period('2021-05-10', 'D'),
                                            Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="AHJZT", start_date=df["date_day_shifted"].min(),
                     end_date=df["date_day_shifted"].max(), live=False)

    with pytest.raises(MissingDataException):
        prices = sp.download()


def test_live_missing_data():
    df = pd.DataFrame({"date_day_shifted": [Period('2021-05-10', 'D'),
                                            Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="AHJZT", start_date=df["date_day_shifted"].min(),
                     end_date=df["date_day_shifted"].max(), live=True)

    with pytest.raises(MissingDataException):
        prices = sp.download()


def test_live():
    sp = StockPrices(ticker_name="AAPL", start_date=Period('2021-05-10', 'D'),
                     end_date=Period('2021-05-20', 'D'), live=True)

    if Period.now("D").weekday < 5 and datetime.datetime.now().hour >= 15:
        prices = sp.download()
        assert prices["date_day_shifted"].loc[len(prices) - 1] == Period.now("D")


def test_space_removal():
    df = pd.DataFrame({"date_day_shifted": [Period('2021-05-10', 'D'),
                                            Period('2021-05-20', 'D')]})

    sp = StockPrices(ticker_name="GME", start_date=df["date_day_shifted"].min(),
                     end_date=df["date_day_shifted"].max(), live=False)
    prices = sp.download()

    assert list(prices.columns) == ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'date_day_shifted']


def test_live_close_start_end():
    sp = StockPrices(ticker_name="AAPL", start_date=Period.now("D") - datetime.timedelta(days=1),
                     end_date=Period('2021-05-20', 'D'), live=True)

    if Period.now("D").weekday < 5 and datetime.datetime.now().hour >= 15:
        prices = sp.download()
        assert prices["date_day_shifted"].loc[len(prices) - 1] == Period.now("D")
