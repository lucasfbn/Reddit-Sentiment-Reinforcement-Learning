import pandas as pd
from pandas.testing import assert_frame_equal

from rl._eval.envs.utils.utils import order_day_wise, ticker_list_to_df
from tests.utils import MockObj

ticker = [
    MockObj(name="1",
            sequences=[MockObj(price_raw=1, date=pd.Period("22/10/21"),
                               tradeable=True, action=1, action_probas={})]),
    MockObj(name="3",
            sequences=[MockObj(price_raw=2, date=pd.Period("26/10/21"),
                               tradeable=True, action=1, action_probas={})]),
    MockObj(name="2",
            sequences=[MockObj(price_raw=3, date=pd.Period("23/10/21"),
                               tradeable=True, action=1, action_probas={}),
                       MockObj(price_raw=4, date=pd.Period("26/10/21"),
                               tradeable=True, action=1, action_probas={})]),
    MockObj(name="4",
            sequences=[MockObj(price_raw=5, date=pd.Period("22/10/21"),
                               tradeable=True, action=1, action_probas={})]),
]


def test_ticker_list_to_df():
    result = ticker_list_to_df(ticker)

    expected = pd.DataFrame({'ticker': ['1', '3', '2', '2', '4'], 'price': [1, 2, 3, 4, 5],
                             'date': [pd.Period('2021-10-22', 'D'), pd.Period('2021-10-26', 'D'),
                                      pd.Period('2021-10-23', 'D'), pd.Period('2021-10-26', 'D'),
                                      pd.Period('2021-10-22', 'D')],
                             'tradeable': [True, True, True, True, True], 'action': [1, 1, 1, 1, 1],
                             'action_probas': [{}, {}, {}, {}, {}], 'ticker_id': [0, 1, 2, 2, 3],
                             'seq_id': [0, 0, 0, 1, 0]}
                            )

    assert_frame_equal(result, expected)


def test_order_daywise():
    df = ticker_list_to_df(ticker)
    result = order_day_wise(ticker, df)

    day_1 = result[pd.Period("22/10/21")]
    assert day_1[0].ticker == "1" and day_1[0].price == 1
    assert day_1[1].ticker == "4" and day_1[1].price == 5

    day_2 = result[pd.Period("23/10/21")]
    assert day_2[0].ticker == "2" and day_2[0].price == 3

    day_3 = result[pd.Period("26/10/21")]
    assert day_3[0].ticker == "3" and day_3[0].price == 2
    assert day_3[1].ticker == "2" and day_3[1].price == 4
