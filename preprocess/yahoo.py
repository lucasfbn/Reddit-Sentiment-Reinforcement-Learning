import yfinance as yf
import pandas as pd
import datetime


def merge(df, symbol, start_offset):
    start = str(df["date_day"].min() - datetime.timedelta(days=start_offset))
    end = str(df["date_day"].max())

    historical_data = yf.download(symbol, start=start, end=end)
    historical_data["date_day"] = pd.to_datetime(historical_data.index).to_period('D')

    if start_offset == 0:
        how = "left"
    else:
        how = "right"

    df = df.merge(historical_data, on="date_day", how=how)
    return df
