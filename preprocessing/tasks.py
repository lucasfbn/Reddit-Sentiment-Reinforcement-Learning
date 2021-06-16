from typing import Tuple

import pandas as pd
from prefect import task
from sklearn.preprocessing import MinMaxScaler

from preprocessing.sequences import FlatSequenceGenerator, ArraySequenceGenerator
from preprocessing.stock_prices import StockPrices, MissingDataException, OldDataException

date_col = "date"
date_day_col = "date_day"
date_shifted_col = "date_shifted"
date_day_shifted_col = "date_day_shifted"


class Ticker:

    def __init__(self, name, df):
        self.df = df
        self.name = name
        self.metadata = None
        self.exclude = False

        self.flat_sequence = None
        self.array_sequence = None


@task
def add_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses and sorts the date.

    Adds:
        [date_col, date_day_col]
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

    Adds:
        [date_shifted_col, date_day_shifted_col]
    """
    df[date_shifted_col] = df[date_col] - pd.Timedelta(hours=start_hour,
                                                       minutes=start_min) + pd.Timedelta(days=1)

    df[date_day_shifted_col] = pd.to_datetime(df['date_shifted']).dt.to_period('D')
    return df


@task
def drop_columns(df: pd.DataFrame, columns_to_be_dropped: list):
    """
    Drops columns from a df.
    """
    return df.drop(columns=columns_to_be_dropped)


def handle_scaled_columns(df: pd.DataFrame, unscaled_cols: list, drop_unscaled_cols: bool) \
        -> Tuple[pd.DataFrame, list]:
    """
    Handles columns after scaling. If drop_unscaled_cols is True, the unscaled columns will be dropped from the df and
    only the scaled columns will be returned as the new columns. Otherwise the scaled columns will be added to the
    unscaled columns and returned

    Args:
        df:
        unscaled_cols: List of columns that got scaled
        drop_unscaled_cols: Whether to drop the raw column (names)
    """

    scaled_cols = [col + "_scaled" for col in unscaled_cols]

    if drop_unscaled_cols:
        df = df.drop(columns=unscaled_cols)
        new_cols = scaled_cols
    else:
        new_cols = unscaled_cols + scaled_cols

    return df, new_cols


@task
def scale_sentiment_data_daywise(df: pd.DataFrame, sentiment_data_cols: list,
                                 drop_unscaled_cols: bool) -> Tuple[pd.DataFrame, list]:
    """
    Scales all columns, which are in cols_to_be_scaled daywise. Therefore, group for the (shifted)
    date prior to scaling.

    Args:
        df:
        sentiment_data_cols: List of columns that shall not be scaled
        drop_unscaled_cols: Whether to drop the raw columns after scaling or not

    Returns:
        The scaled df and the new sentiment data column when drop_unscaled_cols was True. Else the old sentiment data
        columns + the new scaled columns will be returned.
    """
    dates = df.groupby([date_day_shifted_col])

    scaled = []

    for date, date_df in dates:
        date_df = scale(date_df, sentiment_data_cols)
        scaled.append(date_df)

    new_df = pd.concat(scaled)

    new_df, sentiment_data_cols = handle_scaled_columns(new_df, sentiment_data_cols, drop_unscaled_cols)
    return new_df, sentiment_data_cols


def scale(df: pd.DataFrame, cols_to_be_scaled: list):
    scaler = MinMaxScaler()
    df[[col + "_scaled" for col in cols_to_be_scaled]] = scaler.fit_transform(df[cols_to_be_scaled])
    return df


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

    Adds:
        ["available"]

    Returns:
        Same Ticker instance but with added "available" column
    """

    # Assert that the df is ordered correctly. This is important since we mark the first n days as not available
    # (and these should be the the first, not the last or random days)
    assert ticker.df[date_shifted_col].iloc[0] <= ticker.df[date_shifted_col].iloc[len(ticker.df) - 1]

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

    Adds:
        ["Open", "High", "Low", "Close", "Volume", "Adj Close"]

    Args:
        ticker: Ticker instance
        price_data_start_offset: Offset in days from the min_date in the ticker.df. This is useful when you want to add
         certain price information prior to the first occurrence of a ticker in our data
        enable_live_behaviour: Whether to enable the live behaviour. When this is true the max date in the ticker df is
         ignored and the current date will be taken instead. Also, some additional assertions are checked.
         See stock_prices.py for additional informations.
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


@task
def assign_price_col(ticker: Ticker, price_col: str) -> Ticker:
    """
    Assign a certain price (either "Open" or "Close") to a unified "price" column.
    This is just a convenience to avoid having to use the actual used price later on and can instead rely on
    the "price" column being the correct price.

    Args:
        ticker:
        price_col: Used price ("Close", "Open" etc.)

    Adds:
        ["price"]
    """
    ticker.df["price"] = ticker.df[price_col]
    return ticker


@task
def mark_tradeable_days(ticker: Ticker) -> Ticker:
    """
    Marks all days that are no weekend days and, therefore, in theory tradeable. This might by used later on in order to
    avoid trading on days that are not tradeable in reality.

    NOTE: In case you might be wondering why this wasn't done prior to the ticker split: We might have added some
     additional days while running add_price_data

    Args:
        ticker:

    Adds:
        ["tradeable"]

    """

    ticker.df["temp_weekday"] = ticker.df[date_col].dt.dayofweek
    ticker.df["tradeable"] = ticker.df["temp_weekday"] < 5
    ticker.df = ticker.df.drop(columns=["temp_weekday"])
    return ticker


@task
def drop_ticker_df_columns(ticker: Ticker, columns_to_be_dropped: list) -> Ticker:
    """
    Drops columns from a ticker df.
    """
    ticker.df = ticker.df.drop(columns=columns_to_be_dropped)
    return ticker


@task
def forward_fill_price(ticker: Ticker) -> Ticker:
    """
    Forward fills the price. This is useful when we have data from a date where no price data exists (on the weekend
    for example). In such a case the price from friday will be carried forward to saturday and sunday.

    Args:
        ticker:

    Returns:

    """
    ticker.df["price"] = ticker.df["price"].fillna(method="ffill")
    return ticker


@task
def mark_ticker_where_all_prices_are_nan(ticker: Ticker) -> Ticker:
    """
    Marks ticker to exclude when the whole price column is None. The forward fill from the preceding task will not cover
     this case since it would just forward fill with nan.
    """

    if ticker.df["price"].isnull().all():
        ticker.exclude = True
    return ticker


@task
def mark_ipo_ticker(ticker: Ticker) -> Ticker:
    """
    Marks ticker to exclude when, after forward filling and excluding all-nan ticker, there are still nan values in the
     price column. This may occur when - due to IPOs - there are no prices available prior to a certain date (the IPO
     date).

    Args:
        ticker:

    Returns:

    """

    if ticker.df["price"].isnull().any():
        ticker.exclude = True
    return ticker


@task
def fill_missing_sentiment_data(ticker: Ticker, sentiment_data_columns: list) -> Ticker:
    """
    Since we most likely added some nans to our sentiment data (due to add_price_data) we will this missing data with 0.

    Args:
        ticker:
        sentiment_data_columns: Columns with the sentiment data information

    Returns:

    """
    ticker.df[sentiment_data_columns] = ticker.df[sentiment_data_columns].fillna(0)
    return ticker


@task
def assert_no_nan(ticker: Ticker):
    """
    Checks for any nans in the whole ticker df.

    Args:
        ticker:

    Returns:

    """
    assert ticker.df.isnull().values.any() == False


@task
def add_metric_rel_price_change(ticker: Ticker) -> Ticker:
    """
    Adds metric: relative price change in relation to prior day
    Formula:
        rel_change = (price[1] - price[0]) / price[0]

    See test for an example.

    Args:
        ticker:

    Returns:

    """

    prices = ticker.df["price"].tolist()

    rel_change = []
    for i, _ in enumerate(prices):
        if i == 0:
            rel_change.append(0.0)
        else:
            rel_change.append((prices[i] - prices[i - 1]) / (prices[i - 1]))

    ticker.df["price_rel_change"] = rel_change
    return ticker


@task
def remove_old_price_col_from_price_data_columns(price_data_columns: list, price_column: str):
    """
    Removes the used price (mostly "Close") from the price_data list and adds the new price column (which is basically
    the old price column renamed)

    Args:
        price_data_columns: List of price data columns
        price_column: Used price column (like "Close", "Open", ...)
    """
    price_data_columns.remove(price_column)
    price_data_columns += ["price"]
    return price_data_columns


@task
def scale_price_data(ticker: Ticker, price_data_columns: list, drop_unscaled_cols: bool) -> Tuple[Ticker, list]:
    """
    Scales the price data (or any other arbitrage list of columns).

    Args:
        ticker:
        price_data_columns: Column names of the price data
        drop_unscaled_cols:
    """

    ticker.df = scale(ticker.df, cols_to_be_scaled=price_data_columns)
    ticker.df, price_data_columns = handle_scaled_columns(ticker.df, price_data_columns, drop_unscaled_cols)
    return ticker, price_data_columns


@task
def add_metadata_to_ticker(ticker: Ticker, metadata_cols: list) -> Ticker:
    """
    Basically copies part of the df to a separate df. This might be useful later on when the actual df is transformed
    into a timeseries.

    Args:
        ticker:
        metadata_cols:
    """
    ticker.metadata = ticker.df[metadata_cols]
    return ticker


@task
def make_sequences(ticker: Ticker, sequence_length: int, include_available_days_only: bool,
                   columns_to_be_excluded_from_sequences: list, price_column: str) -> Ticker:
    """
    Generates flat and array sequences from a given ticker df. For further details on what sequences are please check
    the documentation in the sequence class (sequences.py) itself.

    Args:
        ticker:
        sequence_length: Length of the desired sequence
        include_available_days_only: Whether to filter sequences which were not available for trading
        columns_to_be_excluded_from_sequences: Cols that may be used during the generation of sequences but are not
         subject of the final sequences and can therefore be dropped
        last_column: The column of each sequence that shall be the last one. This is usually the price column since the
         NN uses the last column as an indicator of the current price.
    """
    flat_seq = FlatSequenceGenerator(df=ticker.df, sequence_len=sequence_length,
                                     include_available_days_only=include_available_days_only,
                                     exclude_cols_from_sequence=columns_to_be_excluded_from_sequences,
                                     price_column=price_column)
    ticker.flat_sequence = flat_seq.make_sequence()

    arr_seq = ArraySequenceGenerator(df=ticker.df, sequence_len=sequence_length,
                                     include_available_days_only=include_available_days_only,
                                     exclude_cols_from_sequence=columns_to_be_excluded_from_sequences,
                                     price_column=price_column)
    ticker.array_sequence = arr_seq.make_sequence()
    return ticker


@task
def mark_empty_sequences(ticker: Ticker):
    """
    Marks the sequences which are empty due to entries not being tradeable or available or being too short.
    """
    assert len(ticker.array_sequence) == len(ticker.flat_sequence)

    if not ticker.array_sequence:
        ticker.exclude = True

    return ticker
