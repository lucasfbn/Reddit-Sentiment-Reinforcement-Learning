import datetime

import pandas as pd
import yfinance as yf


def merge(df, symbol, start_offset):
    start = str(df["date_day"].min() - datetime.timedelta(days=start_offset))
    end = str(df["date_day"].max())

    historical_data = yf.download(symbol, start=start, end=end)
    historical_data["date_day"] = pd.to_datetime(historical_data.index).to_period('D')

    df = historical_data.merge(df, on="date_day", how="outer")
    return df
