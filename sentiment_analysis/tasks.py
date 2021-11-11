import re
from datetime import datetime

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from simplepipeline import task, filter_task

from utils import paths
from sentiment_analysis.reddit_data.api.google_cloud import BigQueryDB, DetectGaps
from sentiment_analysis.timespan import Timespan


@task
def get_from_gc(start: datetime, end: datetime, check_duplicates: bool, fields_to_retrieve: list) -> pd.DataFrame:
    """
    Downloads the raw data from GC.

    Args:
        start: Start date
        end: End date
        check_duplicates: Whether to drop duplicates or not
        fields_to_retrieve: Subset of all possible fields

    Returns:
        Data as DataFrame
    """
    db = BigQueryDB()
    df = db.download(start=start, end=end, fields=fields_to_retrieve, check_duplicates=check_duplicates)
    return df


@task
def retrieve_gaps(df: pd.DataFrame):
    """
    Retrieves the gaps within the data (e.g. hours/day where no data was scraped from reddit).
    """
    db = DetectGaps(df)
    return db.run()


@filter_task
def filter_removed(df: pd.DataFrame, cols_to_check_if_removed: list) -> pd.DataFrame:
    """
    Checks if the submission got removed and filters it if it was.

    Args:
        df: DataFrame
        cols_to_check_if_removed: List of column which are inspected for certain deletion-specific keywords

    Returns:
        Filtered DataFrame
    """
    for col in cols_to_check_if_removed:
        df = df[~df[col].isin(["[removed]", "[deleted]"])]
    return df


@task
def add_temporal_informations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds several informations to the each submission in the DF.

    - converts date to MESZ (timezone), and replace the default date (the origin column "created_utc" is in UTC format)
    - keeps only the day part of the date from above (so, hours etc are ignored - useful when filtering for stuff that
      happened within one day)
    - start (also as timestamp)
    - end (also as timestamp)

    For start and end we refer to the timespan (hour) in which the submission was created.
    For instance: date_mesz: 10/10/2021 06:15
        => start = 10/10/2021 06:00, end = 10/10/2021 07:00

    This is useful/necessary since we will later aggregate the data based on each hour of the covered timeperiod.
    E.g. if we want to work with the data from 01.01.2021 00:00 until 02.01.2021 00:00 we will have 24 start and end
    periods.

    Args:
        df:

    Returns:

    """

    def _get_unix_timestamp(date):
        return (date - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

    date = pd.to_datetime(df["created_utc"], unit="s")
    date = date.dt.tz_localize("UTC")
    df["date"] = date.dt.tz_convert("Europe/Berlin")
    df["date_day"] = df["date"].dt.to_period("D")

    start = df["date"].dt.to_period('H').dt.to_timestamp()
    df["start"] = start
    df["start_timestamp"] = _get_unix_timestamp(start)

    end = (df["date"] + pd.Timedelta(seconds=3600)).dt.to_period('H').dt.to_timestamp()
    df["end"] = end
    df["end_timestamp"] = _get_unix_timestamp(end)
    return df


@filter_task
def filter_authors(df: pd.DataFrame, filter_too_frequent_authors: bool, author_blacklist: list,
                   max_submissions_per_author_per_day: int) -> pd.DataFrame:
    """
    Filters submissions based on a given number of allowed submission per author per day.

    Args:
        df:
        filter_too_frequent_authors: Whether to apply this filter or not
        author_blacklist: List of author names whose submissions are filtered by default
        max_submissions_per_author_per_day: The max number of submissions per author per day

    Returns:

    """

    if not filter_too_frequent_authors:
        return df

    filtered_rows = []

    cols = df.columns

    # Group by the submissions of a single day
    days = df.groupby(["date_day"])

    for day, submissions in days:
        submissions = pd.DataFrame(submissions)

        # Group by all authors within the day
        authors = submissions.groupby(["author"])

        for author_id, author_submissions in authors:

            if author_id in author_blacklist:
                continue

            if len(author_submissions) <= max_submissions_per_author_per_day:
                filtered_rows.append(author_submissions)
            else:
                # If the number of submissions per author per day exceeds the threshold:
                # Sort by number of comment and only take the n-top ones
                # (e.g. drop the rest of the submissions from the same author)
                author_submissions = author_submissions.sort_values(by=["num_comments"], ascending=False)
                filtered_rows.append(author_submissions.head(max_submissions_per_author_per_day))

    df = pd.concat(filtered_rows)
    df.columns = cols
    return df


@task
def delete_non_alphanumeric(df, cols_to_be_cleaned: list) -> pd.DataFrame:
    """
    Removes all non alphanumeric characters.
    We do NOT remove stuff like "!, ?, :), :D" since it adds value to the sentiment analysis later on.

    Args:
        df:
        cols_to_be_cleaned: Columns to be cleaned

    Returns:

    """

    for col in cols_to_be_cleaned:
        df[col] = df[col].str.replace('[^\w\s,.?!()-+:"]', '', regex=True)
    return df


@task
def load_valid_ticker(valid_ticker_path):
    ticker = pd.read_csv(paths.all_ticker, sep=";")["Symbol"]
    ticker = ticker[ticker.str.len() >= 2]
    return ticker.values.tolist()


"""
Regex compiled outside of method so it is only called (compiled) once
Explanation:
(?<=\W)|(^)) - look behind whether the character prior to the current one was not a word and not the 
 start of the line
[A-Z]{2,5} - match any uppercase word with length between 2 and 5 characters 
(?=(\W|$)) - look ahead whether the character after the current one is not a word and not the end of the line

For further informations on look behind/ahead:
 https://stackoverflow.com/questions/2973436/regex-lookahead-lookbehind-and-atomic-groups
"""
ticker_regex = re.compile(r"((?<=\W)|(^))[A-Z]{2,5}(?=(\W|$))", re.MULTILINE)


def extract_ticker(txt: str, valid_ticker: list, ticker_blacklist: list):
    """
    Extracts (stock) ticker from a given string. Each ticker has to be in the valid_ticker list and will be ignored
    if in the ticker_blacklist.

    Args:
        txt: str with potential ticker
        valid_ticker: List of valid ticker
        ticker_blacklist: List of ticker that shall be ignored (even tho they might be in valid_ticker)

    Returns:
        A list of all ticker that have been found in the txt
    """

    # TODO We could replace "word not in occured_ticker" by list(set(occured_ticker)) before return occured_ticker
    # However, this changes order from time to time and thus the tests will fail. Also not sure if this improves
    # performance. It's probably rather unimportant.

    occurred_ticker = []

    # Case when nan for instance
    if not isinstance(txt, str):
        return None

    potential_ticker = []

    occurred_ticker = []
    for match in ticker_regex.finditer(txt):
        potential_ticker.append(match.group())

    for word in potential_ticker:
        if word not in ticker_blacklist and word in valid_ticker and word not in occurred_ticker:
            occurred_ticker.append(word)

    if not occurred_ticker:
        return None
    return occurred_ticker


@task
def get_submission_ticker(df: pd.DataFrame, valid_ticker: list, ticker_blacklist: list,
                          search_ticker_in_body: bool) -> pd.DataFrame:
    """
    Retrieves all ticker within a submission. A potential ticker has to be in valid_ticker and not in ticker_blacklist.
    Also, if the search should be extended to the submission body (default: only title is searched),
    search_ticker_in_body has to be enabled.

    Args:
        df:
        valid_ticker: List of valid ticker
        ticker_blacklist: List of blacklisted ticker
        search_ticker_in_body: Whether to search the submission body or not. Even if True it will only search for ticker
         in the body if there were no ticker found in the title

    Returns:

    """
    df["title_ticker"] = df.apply(lambda row: extract_ticker(row["title"], valid_ticker, ticker_blacklist),
                                  axis="columns")

    if search_ticker_in_body:
        df["body_ticker"] = df.apply(lambda row: (extract_ticker(row["selftext"], valid_ticker, ticker_blacklist)
                                                  if row["title_ticker"] is None else None), axis="columns")
    else:
        df["body_ticker"] = None

    # Currently, the body of a submission shall not be searched for ticker when there was a valid ticker in the title
    assert len(df[(df["title_ticker"].notnull()) & (df["body_ticker"].notnull())]) == 0

    return df


@filter_task
def filter_submissions_without_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters all submission without a valid ticker in either the title or the body.

    Logic:
    Filter all submissions where ¬(¬title_ticker ∧ ¬body_ticker)

    Truth Table
    body_ticker	title_ticker	¬(¬title_ticker ∧ ¬body_ticker)
    F	F	                                    F
    F	T	                                    T
    T	F	                                    T
    T	T	                                    T

    (Although the last entry of the TT is currently not possible, see @get_submission_ticker)

    Args:
        df:

    Returns:

    """
    df = df[~((df["title_ticker"].isnull()) & (df["body_ticker"].isnull()))]
    return df


@task
def merge_ticker_to_a_single_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the two columns "title_ticker" and "body_ticker" into one column, depending in their respective value.
    The two columns "title_ticker" and "body_ticker" are dropped afterwards.

    Examples:

    | title_ticker | body_ticker | ticker |
    |--------------|-------------|--------|
    | GME          | None        | GME    |
    | None         | GME         | GME    |

    Args:
        df:

    Returns:

    """
    df["ticker"] = df["title_ticker"].fillna(df["body_ticker"])
    df = df.drop(columns=["title_ticker", "body_ticker"])
    return df


@task
def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic sentiment analyzer. Analyses the sentiment of each **title** of each submission.
    Currently, each submission is extended by an estimation how much of the text is positive, neutral or negative.
    An additional compound score is issued, indicating the overall score of the text.

    # TODO Analyze body, not only title.
    # TODO Analyze comments?

    Args:
        df:

    Returns:

    """
    analyzer = SentimentIntensityAnalyzer()

    def _analyze(txt):
        scores = analyzer.polarity_scores(txt)
        return [scores["pos"], scores["neu"], scores["neg"], scores["compound"]]

    df[["pos", "neu", "neg", "compound"]] = df.apply(lambda row: _analyze(row["title"]), axis="columns",
                                                     result_type="expand")
    return df


@task
def flatten_ticker_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens the ticker that occur in the submissions. At this point we don't care about the actual
    submission, all we care about is the respective (vader) score and the respective ticker that belong to that text.
    We, therefore, flatten the list of tickers and duplicate the row of a single submission for each ticker.

    Example:

        | ticker      | id |
        |-------------|----|
        | [GME, TSLA] | 1  |
        | [GME]       | 2  |

        gets flattened to:

        | ticker | id |
        |--------|----|
        | GME    | 1  |
        | TSLA   | 1  |
        | GME    | 2  |

    Args:
        df:

    Returns:
        Flattened df.
    """
    flattened = []

    def flatten(row):
        for ticker in row["ticker"]:
            temp = pd.DataFrame(row).T
            temp["ticker"] = ticker
            flattened.append(temp)

    df.apply(flatten, axis="columns")

    flattened_df = pd.concat(flattened)
    return flattened_df


@task
def retrieve_timespans(df: pd.DataFrame, relevant_timespan_cols: list) -> list:
    """
    Groups the submissions/ticker into their respective start and end date periods.
    Only keeps the columns of each submission that are specified in relevant_timespan_cols.

    Args:
        df:
        relevant_timespan_cols: List of columns that shall not be dropped (e.g. are of use later on)

    Returns:
        List of Timespan objects
    """

    timespan_start = df.groupby("start")

    timespans = []
    for timespan, submissions in timespan_start:
        # since the periods is the same within the grouped timespans we can assign start and end of the whole timespan
        # from the first element in the df.
        timespans.append(Timespan(start=submissions["start"].iloc[0],
                                  end=submissions["end"].iloc[0],
                                  df=submissions[relevant_timespan_cols]))

    return timespans


@task
def aggregate_submissions_per_timespan(ts: Timespan):
    """
    Aggregates the data of a single timespan (1 hour atm).
    E.g. each pos/neg/neu/compound score gets summed up for each ticker.
    See tests for further insights.
    We also add the number of mentions for each ticker.
    Note: aggregating will drop all columns that are not in [int, float].

    Args:
        ts:

    Adds:
        ["num_posts"]
    """

    # In case some numeric columns are not float or int dtypes (most likely they are object type)
    ts.df = ts.df.apply(pd.to_numeric, errors="ignore")

    # By adding a col where each value is one we can easily see the sum of posts that got aggregated
    # afterwards
    ts.df["num_posts"] = 1

    ts.df = ts.df.groupby("ticker").agg("sum").reset_index()
    return ts


@task
def summarize_timespans(timespans: list) -> pd.DataFrame:
    """
    "Summarizes" all timespans by flattening them. Each ticker in each timespan will be added to a separate DF,
    containing the aggregated values and its start and end date.

    Args:
        timespans: List of Timespan objects

    Returns:
        Flattened DF of the timespans
    """

    dfs = []

    for timespan in timespans:
        df = timespan.df
        df["start"] = timespan.start
        df["end"] = timespan.end
        dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)
