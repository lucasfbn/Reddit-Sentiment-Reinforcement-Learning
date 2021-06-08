import pandas as pd
from prefect import task
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from preprocessing.logic.stock_prices import StockPrices, MissingDataException, OldDataException

date_col = "date"
date_day_col = "date_day"
date_shifted_col = "date_shifted"
date_day_shifted_col = "date_day_shifted"


class Ticker:

    def __init__(self, name, df):
        self.df = df
        self.name = name
        self.exclude = False


@task
def add_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses and sorts the date.

    Args:
        df:

    Returns:

    """
    df[date_col] = pd.to_datetime(df["end"], format="%Y-%m-%d %H:%M")
    df[date_day_col] = df[date_col].dt.to_period('D')
    df = df.sort_values(by=[date_col])
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
    df[date_shifted_col] = df[date_col] - pd.Timedelta(hours=start_hour,
                                                       minutes=start_min) + pd.Timedelta(days=1)

    df[date_day_shifted_col] = pd.to_datetime(df['date_shifted']).dt.to_period('D')
    return df


@task
def get_min_max_time(df: pd.DataFrame) -> Tuple[pd.Period, pd.Period]:
    return df[date_day_shifted_col].min(), df[date_day_shifted_col].max()


@task
def scale_daywise(df: pd.DataFrame, cols_to_be_scaled: list, drop_scaled_cols: bool) -> pd.DataFrame:
    """
    Scales all columns, which are in cols_to_be_scaled daywise. Therefore, group for the (shifted)
    date prior to scaling.

    Args:
        df:
        cols_to_be_scaled: List of columns that shall not be scaled
        drop_scaled_cols: Whether to drop the raw columns after scaling or not

    Returns:

    """
    dates = df.groupby([date_day_shifted_col])

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
    Groups ticker, created a new Ticker instance for each.

    Args:
        df:

    Returns:
        List of Ticker instances for each ticker in df
    """

    ticker = []

    for name, ticker_df in df.groupby(["ticker"]):
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
def sort_ticker_df_chronologically(ticker: Ticker) -> Ticker:
    ticker.df = ticker.df.sort_values(by=[date_shifted_col], ascending=True)
    return ticker


@task
def mark_trainable_days(ticker: Ticker, ticker_min_len: int) -> Ticker:
    """
    Marks the days within a ticker df that were not tradeable.
    For instance, if we set min_len to 2, we, in reality, are not able to trade the instances prior to the entry
    which fulfills the min len.
    The ticker df must be sorted - newest entry at the bottom!

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
    # (and these should be the the first, not the last or random days)
    assert ticker.df.loc[0, "date_shifted"] <= ticker.df.loc[len(ticker.df) - 1, "date_shifted"]

    # Create available series
    available = [False] * ticker_min_len + [True] * (len(ticker.df) - ticker_min_len)
    # Add series to df
    ticker.df["available"] = available

    return ticker


@task
def add_price_data(ticker: Ticker, price_data_start_offset: int, enable_live_behaviour: bool) -> Ticker:
    """
    Adds price data to a ticker df. Be aware that is merges the "outer" values, meaning that gaps
    between two dates will be filled with price data (as long as there is any at the specific day).
    If any price related exception occurs the ticker will be marked to be excluded (in one of the next tasks).

    Args:
        ticker: Ticker instance
        price_data_start_offset: Offset in days from the min_date in the ticker.df. This is useful when you want to add
         certain price information prior to the first occurrence of a ticker in our data
        enable_live_behaviour: Whether to enable the live behaviour. When this is true the max date in the ticker df is
         ignored and the current date will be taken instead. Also, some additional assertions are checked.
         See stock_prices.py for additional informations.

    Returns:

    """

    sp = StockPrices(ticker_name=ticker.name, ticker_df=ticker.df, start_offset=price_data_start_offset,
                     live=enable_live_behaviour)

    try:
        prices = sp.download()
        merged = sp.merge()
        ticker.df = merged

    # TODO Add logging
    except (MissingDataException, OldDataException) as e:
        ticker.exclude = True

    return ticker


@task
def remove_excluded_ticker(ticker: list) -> list:
    """
    Removes all ticker that are marked for exclusion.

    Args:
        ticker: List of Ticker instances

    Returns:
        Filtered list of Ticker instances
    """

    valid_ticker = []

    for t in ticker:
        if not t.exclude:
            valid_ticker.append(t)

    return valid_ticker


@task
def backfill_availability(ticker: Ticker) -> Ticker:
    """
    Refills the "available" column since we might have added some nans to our sentiment data by merging with the
     price data (see add_price_data).

    Example:
        Prior to add_price_data

        date        compound    available
        05/01/2021  10          False
        07/01/2021  20          True

        After add_price_data

        date        compound    available
        05/01/2021  10          False
        06/01/2021  nan         nan
        10/01/2021  20          True

        Expected afterwards:

        date        compound    available
        05/01/2021  10          False
        06/01/2021  nan         False
        10/01/2021  20          True

        available


    Args:
        ticker:

    Returns:

    """
    first_availability = ticker.df["available"].iloc[0]

    # If either False or nan
    if first_availability != True:
        # Set False so we can proceed to forward fill from there
        ticker.df["available"].iloc[0] = False

    # For details on how forward fill works: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ffill.html
    ticker.df["available"] = ticker.df["available"].fillna(method="ffill")
    return ticker


def rename_cols(ticker: Ticker) -> Ticker:
    pass
