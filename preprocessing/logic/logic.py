from datetime import datetime

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from prefect import task
from typing import Tuple
import paths
from mlflow_api import log_file
from sentiment_analysis.logic.timespan import Timespan
from sentiment_analysis.reddit_data.api.google_cloud import BigQueryDB
from sklearn.preprocessing import MinMaxScaler


class Ticker:

    def __init__(self, name, df):
        self.df = df
        self.name = name


@task
def add_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses and sorts the date.

    Args:
        df:

    Returns:

    """
    df["date"] = pd.to_datetime(df["end"], format="%Y-%m-%d %H:%M")
    df["date_day"] = df["date"].dt.to_period('D')
    df = df.sort_values(by=["date"])
    return df


@task
def shift_time(df: pd.DataFrame, start_hour: int, start_min: int) -> pd.DataFrame:
    """
    Shifts all dates in the df by a given hour and given minutes plus one day.

    Example:
    Suppose we want to trade at 08:00 a.m., we shift all dates by 8 hours (plus 1 day).
    Then, we strip the hour of and only keep the day:

    Original                         Transformed
    --------                         -----------
    04.01.2021 08:00 / 04.01.2021 -> 05.01.2021 00:00 / 05.01.2021
    04.01.2021 09:00 / 04.01.2021 -> 05.01.2021 01:00 / 05.01.2021
    ...
    05.01.2021 07:00 / 05.01.2021 -> 05.01.2021 23:00 / 05.01.2021
    05.01.2021 08:00 / 05.01.2021 -> 06.01.2021 00:00 / 06.01.2021

    We have now achieved that 04.01.2021 08:00 is now counted as 05.01.2021 because it is within the 24h range from
    05.01.2021 08:00 on. Prior to our transformation it was treated as a different day.

    So, basically we reassigned the days based on whether a given date is within a 24h range of a given start hour or
    not.

    Args:
        df:
        start_hour: A given hour at which to start a 24h observation range
        start_min: A given minute at which to start a 24h observation range

    Returns:

    """
    df["date_shifted"] = df["date"] - pd.Timedelta(hours=start_hour,
                                                   minutes=start_min) + pd.Timedelta(days=1)

    df["date_shifted_day"] = pd.to_datetime(df['date_shifted']).dt.to_period('D')
    return df


@task
def get_min_max_time(df: pd.DataFrame) -> Tuple[pd.Period, pd.Period]:
    return df["date_shifted_day"].min(), df["date_shifted_day"].max()


@task
def scale_daywise(df: pd.DataFrame, excluded_cols_from_scaling: list, drop_scaled_cols: bool) -> pd.DataFrame:
    """
    Scales all columns, which are not excluded in excluded_cols_from_scaling daywise. Therefore, group for the (shifted)
    date prior to scaling.

    Args:
        df:
        excluded_cols_from_scaling: List of columns that shall not be scaled
        drop_scaled_cols: Whether to drop the raw columns after scaling or not

    Returns:

    """
    cols_to_be_scaled = [x for x in df.columns if x not in excluded_cols_from_scaling]

    dates = df.groupby(["date_shifted_day"])

    scaler = MinMaxScaler()
    scaled = []

    for date, date_df in dates:
        date_df[[col + "_scaled" for col in cols_to_be_scaled]] = scaler.fit_transform(date_df[cols_to_be_scaled])
        scaled.append(date_df)

    new_df = pd.concat(scaled)

    if drop_scaled_cols:
        new_df = new_df.drop(columns=cols_to_be_scaled)

    return new_df


@task
def grp_by_ticker(df: pd.DataFrame) -> list:
    """
    Groups ticker, created a new Ticker instance for each. The ticker df will be sorted - the newest entry will be at
    the bottom (This is required by later tasks).

    Args:
        df:

    Returns:
        List of Ticker instances for each ticker in df
    """

    ticker = []

    for name, ticker_df in df.groupby(["ticker"]):
        ticker_df = ticker_df.sort_values(by=["date_shifted"], ascending=True)
        ticker.append(Ticker(name=name, df=ticker_df))

    return ticker


@task
def drop_ticker_with_too_few_data(ticker: list, ticker_min_len: int) -> list:
    """
    Drops all ticker which have too few data points (represented by the len of the corresponding df). This is useful when
    you want to have a minimum number of days in which the corresponding ticker was mentioned (and therefore is present
    in our data). Ticker with a low number of days mentioned might by subject to bad ticker. Also, a ticker with only
    one entry is not usable during training and will therefore not be recognized by the model. A minimum ticker len of
    2 is strongly recommended.

    Args:
        ticker: List of Ticker instances
        ticker_min_len: The minimum number of entries for each ticker

    Returns:
        Filtered list of the ticker argument
    """

    filtered_ticker = []

    for t in ticker:
        if len(t.df) < ticker_min_len:
            continue
        filtered_ticker.append(t)

    return filtered_ticker


@task
def mark_trainable_days(ticker: Ticker, ticker_min_len: int) -> Ticker:
    """
    Marks the days within a ticker df that were not tradeable.
    For instance, if we set min_len to 2, we, in reality, are not able to trade the instances prior to the entry
    which fulfills the min len.

    Example:

        ticker_min_len = 3
        04.01 - first entry, NOT tradeable
        05.01 - second entry, NOT tradeable
        06.01 - third entry, IS tradeable

    We will use this later on to be sure to only use tradeable days during training of the network. Otherwise, the
    network would learn patterns of days that would've been filtered by our @drop_ticker_with_too_few_data function and
    this could lead to unrealistic profits.


    Args:
        ticker: Ticker instance
        ticker_min_len: The minimum number of entries for each ticker

    Returns:
        Same Ticker instance but with added "available" column
    """

    # Assert that the df is ordered correctly. This is important since we mark the first n days as not available
    assert ticker.df.loc[0, "date_shifted"] <= ticker.df.loc[len(ticker.df) - 1, "date_shifted"]
    # Create available series
    available = [False] * ticker_min_len + [True] * (len(ticker.df) - ticker_min_len)
    # Add series to df
    ticker.df["available"] = available

    return ticker
