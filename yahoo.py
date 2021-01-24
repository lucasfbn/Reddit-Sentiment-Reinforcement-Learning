import yfinance as yf
import pandas as pd


def merge(ticker, symbol):
    start = str(ticker["date_day"].min())
    end = str(ticker["date_day"].max())

    historical_data = yf.download(symbol, start=start, end=end)
    historical_data["date_day"] = pd.to_datetime(historical_data.index).to_period('D')

    ticker = ticker.merge(historical_data, on="date_day", how="left")
    return ticker
