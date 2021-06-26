from pandas import Timestamp, Period
from pandas.testing import assert_frame_equal

from sentiment_analysis.tasks import *


def test_get_from_gc():
    pass


def test_filter_removed():
    df = pd.DataFrame({"test_0": ["[removed]", "[deleted]", "valid"], "test_1": [1, 2, 3]})
    result = filter_removed.run(df, cols_to_check_if_removed=["test_0"]).reset_index(drop=True)
    expected = pd.DataFrame({"test_0": ["valid"], "test_1": [3]}).reset_index(drop=True)
    assert_frame_equal(result, expected, check_index_type=False)


def test_add_temporal_informations():
    df = pd.DataFrame({"created_utc": [1622057400]})
    result = add_temporal_informations.run(df).reset_index(drop=True)
    expected = pd.DataFrame(
        {'created_utc': 1622057400, 'date': Timestamp('2021-05-26 21:30:00+0200', tz='Europe/Berlin'),
         'date_day': Period('2021-05-26', 'D'), 'start': Timestamp('2021-05-26 21:00:00'),
         'start_timestamp': 1622062800, 'end': Timestamp('2021-05-26 22:00:00'), 'end_timestamp': 1622066400},
        index=[0]).reset_index(drop=True)
    assert_frame_equal(result, expected)


def test_filter_authors():
    df = pd.DataFrame({"author": ["test_1", "test_2", "test_3", "blacklisted_author",
                                  "author_with_too_many_subm", "author_with_too_many_subm",
                                  "author_with_too_many_subm"],
                       "num_comments": [10, 5, 0, 9, 66, 55, 44],
                       "date_day": [Period('2021-05-26', 'D')] * 7})

    # Test handling when filter_too_frequent_authors is False
    assert_frame_equal(filter_authors.run(df, False, [], 0), df.copy())

    # Test blacklist - tests whether blacklisted authors are filtered, therefore, set max_submissions_per_author_per_day
    # higher than the highest amount of submissions per author and add an dummy author to the blacklist
    result = filter_authors.run(df, True, author_blacklist=["blacklisted_author"],
                                max_submissions_per_author_per_day=5)
    expected = df[df["author"] != "blacklisted_author"]  # Filter the author that is blacklisted
    assert_frame_equal(result, expected, check_like=True)  # check_list = True since order might differ (not relevant)

    # Test max_submissions
    #   expected behaviour: the submissions of authors, where the number of total submissions is higher than
    #   max_submissions_per_author_per_day should be reduced to max_submissions_per_author_per_day (e.g. only those
    #   submissions that fit into max_submissions_per_author_per_day and have the highest number of comments should be
    #   kept
    result = filter_authors.run(df, True, author_blacklist=[],
                                max_submissions_per_author_per_day=2)
    # Submission with lowest n_comments from author_with_too_many_subm should be filtered
    expected = df[df["num_comments"] != 44]
    assert_frame_equal(result, expected, check_like=True)


def test_delete_non_alphanumeric():
    df = pd.DataFrame({"text": ["text", "text?", "text!", "text12345", "%/§%(&§%(alphanumeric", ":):D"],
                       "ignore": [0, 1, 2, 3, 4, 5]})
    result = delete_non_alphanumeric.run(df, cols_to_be_cleaned=["text"])
    expected = pd.DataFrame(
        [{'text': 'text', 'ignore': 0}, {'text': 'text?', 'ignore': 1}, {'text': 'text!', 'ignore': 2},
         {'text': 'text12345', 'ignore': 3}, {'text': '((alphanumeric', 'ignore': 4}, {'text': ':):D', 'ignore': 5}])
    assert_frame_equal(result, expected)


def test_extract_ticker():
    valid_ticker = ["GME", "TSLA", "AAPL"]

    txt = "GME TSLA AAPL"
    # Convert expected to list(set()) since order might be changed in extract_ticker
    assert extract_ticker(txt, valid_ticker=valid_ticker, ticker_blacklist=[]) == ["GME", "TSLA", "AAPL"]

    txt = "GME, TSLA, AAPL"
    assert extract_ticker(txt, valid_ticker=valid_ticker, ticker_blacklist=[]) == ['GME', 'TSLA', 'AAPL']

    txt = "GME,TSLA,AAPL"
    assert extract_ticker(txt, valid_ticker=valid_ticker, ticker_blacklist=[]) == ['GME', 'TSLA', 'AAPL']

    txt = "GME&TSLA&AAPL"
    assert extract_ticker(txt, valid_ticker=valid_ticker, ticker_blacklist=[]) == ['GME', 'TSLA', 'AAPL']

    txt = "GME & TSLA & AAPL"
    assert extract_ticker(txt, valid_ticker=valid_ticker, ticker_blacklist=[]) == ["GME", "TSLA", "AAPL"]

    # Assert that multiple entries are reduced to one
    txt = "GME GME GME"
    assert extract_ticker(txt, valid_ticker=valid_ticker, ticker_blacklist=[]) == ['GME']

    # Check numeric
    txt = "GME 1TSLA1 1AAPL1"
    assert extract_ticker(txt, valid_ticker=valid_ticker, ticker_blacklist=[]) == ["GME"]

    # Check blacklist
    txt = "GME TSLA AAPL BLK"
    assert extract_ticker(txt, valid_ticker=valid_ticker, ticker_blacklist=["BLK"]) == ["GME", "TSLA", "AAPL"]

    txt = "GMETSLAAAPL"
    assert extract_ticker(txt, valid_ticker=valid_ticker, ticker_blacklist=[]) == None

    txt = "1GME1TSLA1AAPL1"
    assert extract_ticker(txt, valid_ticker=valid_ticker, ticker_blacklist=[]) == None

    txt = "1GME1 TSLA1 AAPL1"
    assert extract_ticker(txt, valid_ticker=valid_ticker, ticker_blacklist=[]) == None


def test_get_submission_ticker():
    valid_ticker = ["GME", "AAPL", "TSLA", "BLK"]

    # Case 1: Find valid ticker in title only
    df = pd.DataFrame({"title": ["GME AAPL", "TSLA", "INVALID"], "selftext": ["GME", "INVALID", "GME"]})
    result = get_submission_ticker.run(df, valid_ticker=valid_ticker, ticker_blacklist=[], search_ticker_in_body=False)
    expected = pd.DataFrame({"title": ["GME AAPL", "TSLA", "INVALID"], "selftext": ["GME", "INVALID", "GME"],
                             "title_ticker": [["GME", "AAPL"], ["TSLA"], None], "body_ticker": [None, None, None]})
    assert_frame_equal(result, expected)

    # Case 2: Ignore blacklisted (but valid) ticker
    df = pd.DataFrame({"title": ["GME AAPL", "TSLA", "INVALID", "BLK"], "selftext": ["GME", "INVALID", "GME", "GME"]})
    ticker_blacklist = ["BLK"]
    result = get_submission_ticker.run(df, valid_ticker=valid_ticker, ticker_blacklist=ticker_blacklist,
                                       search_ticker_in_body=False)
    expected = pd.DataFrame({"title": ["GME AAPL", "TSLA", "INVALID", "BLK"],
                             "selftext": ["GME", "INVALID", "GME", "GME"],
                             "title_ticker": [["GME", "AAPL"], ["TSLA"], None, None],
                             "body_ticker": [None, None, None, None]})
    assert_frame_equal(result, expected)

    # Case 3: Find valid ticker in body if no valid ticker was found in title
    df = pd.DataFrame({"title": ["GME AAPL", "TSLA", "INVALID"], "selftext": ["GME", "INVALID", "GME"]})
    result = get_submission_ticker.run(df, valid_ticker=valid_ticker, ticker_blacklist=[], search_ticker_in_body=True)
    expected = pd.DataFrame({"title": ["GME AAPL", "TSLA", "INVALID"], "selftext": ["GME", "INVALID", "GME"],
                             "title_ticker": [["GME", "AAPL"], ["TSLA"], None], "body_ticker": [None, None, ["GME"]]})
    assert_frame_equal(result, expected)


def test_filter_submissions_without_ticker():
    df = pd.DataFrame({"title_ticker": [["GME"], None, None], "body_ticker": [None, None, ["GME"]]})
    result = filter_submissions_without_ticker.run(df)
    expected = pd.DataFrame({"title_ticker": [["GME"], None], "body_ticker": [None, ["GME"]]})
    assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))


def test_merge_ticker_to_a_single_column():
    df = pd.DataFrame({"title_ticker": [["GME"], None, None], "body_ticker": [None, ["TSLA"], ["GME"]]})
    result = merge_ticker_to_a_single_column.run(df)
    expected = pd.DataFrame({"ticker": [["GME"], ["TSLA"], ["GME"]]})
    assert_frame_equal(result, expected)


def test_analyze_sentiment():
    df = pd.DataFrame({"title": ["VADER is smart, handsome, and funny.",
                                 "At least it isn't a horrible book.",
                                 "The book was good."]})
    result = analyze_sentiment.run(df)
    expected = pd.DataFrame(
        [{'title': 'VADER is smart, handsome, and funny.', 'pos': 0.746, 'neu': 0.254, 'neg': 0.0, 'compound': 0.8316},
         {'title': "At least it isn't a horrible book.", 'pos': 0.363, 'neu': 0.637, 'neg': 0.0, 'compound': 0.431},
         {'title': 'The book was good.', 'pos': 0.492, 'neu': 0.508, 'neg': 0.0, 'compound': 0.4404}], index=[0, 1, 2])
    assert_frame_equal(result, expected)


def test_flatten_ticker_scores():
    df = pd.DataFrame({"ticker": [["GME", "TSLA"], ["TSLA"], ["GME"]], "id": [0, 1, 2]})
    result = flatten_ticker_scores.run(df)
    expected = pd.DataFrame([{'ticker': 'GME', 'id': 0}, {'ticker': 'TSLA', 'id': 0}, {'ticker': 'TSLA', 'id': 1},
                             {'ticker': 'GME', 'id': 2}])
    # dtype changes since we move from list to str for column "ticker"
    assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)


def test_retrieve_timespans():
    df = pd.DataFrame({"submissions": ["subm_1", "subm_2"],
                       "start": [Timestamp('2021-05-26 21:00:00'),
                                 Timestamp('2021-05-26 22:00:00')],
                       "end": [Timestamp('2021-05-26 22:00:00'),
                               Timestamp('2021-05-26 23:00:00')]})
    result = retrieve_timespans.run(df, relevant_timespan_cols=["submissions"])

    # Expected: each "submission" should be in its own Timespan object since they are not within the same
    # start/end period
    ts_1_df = pd.DataFrame({"submissions": ["subm_1"]})
    ts_1 = Timespan(df=ts_1_df, start=Timestamp('2021-05-26 21:00:00'), end=Timestamp('2021-05-26 22:00:00'))

    assert result[0] == ts_1

    ts_2_df = pd.DataFrame({"submissions": ["subm_2"]})
    ts_2 = Timespan(df=ts_2_df, start=Timestamp('2021-05-26 22:00:00'), end=Timestamp('2021-05-26 23:00:00'))
    assert result[1] == ts_2


def test_aggregate_submissions_per_timespan():
    df = pd.DataFrame({"ticker": ["GME", "GME", "GME", "TSLA", "TSLA", "EMA"], "pos": [0.5, 0.5, 0.5, 7.0, 7.0, 10.0],
                       "will_be_dropped": ["yes", "yes", "yes", "yes", "yes", "yes"]})
    ts = Timespan(df=df, start=None, end=None)
    result = aggregate_submissions_per_timespan.run(ts)
    expected = pd.DataFrame([{'ticker': 'EMA', 'pos': 10.0, 'num_posts': 1},
                             {'ticker': 'GME', 'pos': 1.5, 'num_posts': 3},
                             {'ticker': 'TSLA', 'pos': 14.0, 'num_posts': 2}])
    assert_frame_equal(result.df, expected)


def test_summarize_timespans():
    ts_1_df = pd.DataFrame({"submissions": ["subm_1"]})
    ts_1 = Timespan(df=ts_1_df, start=Timestamp('2021-05-26 21:00:00'), end=Timestamp('2021-05-26 22:00:00'))

    ts_2_df = pd.DataFrame({"submissions": ["subm_2"]})
    ts_2 = Timespan(df=ts_2_df, start=Timestamp('2021-05-26 22:00:00'), end=Timestamp('2021-05-26 23:00:00'))

    result = summarize_timespans.run([ts_1, ts_2])
    expected = pd.DataFrame([
        {'submissions': 'subm_1', 'start': Timestamp('2021-05-26 21:00:00'), 'end': Timestamp('2021-05-26 22:00:00')},
        {'submissions': 'subm_2', 'start': Timestamp('2021-05-26 22:00:00'), 'end': Timestamp('2021-05-26 23:00:00')}])
    assert_frame_equal(result, expected)
