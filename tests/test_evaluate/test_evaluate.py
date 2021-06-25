from eval.evaluate import Evaluate
import pandas as pd
from preprocessing.sequences import Sequence
from preprocessing.tasks import Ticker
from pandas.testing import assert_frame_equal
import copy
import pytest

t1 = Ticker("TSLA", None)
t2 = Ticker("AAPL", None)

# Period format: MM-DD-YYYY
t1.sequences = [
    Sequence(price=12, tradeable=True, available=False, date=pd.Period("01-01-2021"), sentiment_data_available=True,
             df=None, price_raw=120,
             flat=pd.DataFrame({"dummy/0": [1], "dummy/1": [2], "dummy/2": [3],
                                "price/0": [10], "price/1": [11], "price/2": [12]}),
             arr=pd.DataFrame({"dummy": [1, 2, 3], "price": [10, 11, 12]})),
    Sequence(price=13, tradeable=True, available=False, date=pd.Period("01-02-2021"), sentiment_data_available=True,
             df=None, price_raw=130,
             flat=pd.DataFrame({"dummy/1": [2], "dummy/2": [3], "dummy/3": [4],
                                "price/1": [11], "price/2": [12], "price/3": [13]}),
             arr=pd.DataFrame({"dummy": [2, 3, 4], "price": [11, 12, 13]})),
    Sequence(price=14, tradeable=True, available=False, date=pd.Period("01-03-2021"), sentiment_data_available=True,
             df=None, price_raw=140,
             flat=pd.DataFrame({"dummy/2": [3], "dummy/3": [4], "dummy/4": [5],
                                "price/2": [12], "price/3": [13], "price/4": [14]}),
             arr=pd.DataFrame({"dummy": [3, 4, 5], "price": [12, 13, 14]}))]

t2.sequences = [
    Sequence(price=12, tradeable=True, available=False, date=pd.Period("01-15-2021"), sentiment_data_available=True,
             df=None, price_raw=120,
             flat=pd.DataFrame({"dummy/0": [81], "dummy/1": [82], "dummy/2": [83],
                                "price/0": [810], "price/1": [811], "price/2": [812]}),
             arr=pd.DataFrame({"dummy": [81, 82, 83], "price": [810, 811, 812]})),
    Sequence(price=13, tradeable=True, available=False, date=pd.Period("01-16-2021"), sentiment_data_available=True,
             df=None, price_raw=130,
             flat=pd.DataFrame({"dummy/1": [82], "dummy/2": [83], "dummy/3": [84],
                                "price/1": [811], "price/2": [812], "price/3": [813]}),
             arr=pd.DataFrame({"dummy": [82, 83, 84], "price": [811, 812, 813]})),
    Sequence(price=14, tradeable=True, available=False, date=pd.Period("01-17-2021"), sentiment_data_available=True,
             df=None, price_raw=140,
             flat=pd.DataFrame({"dummy/2": [83], "dummy/3": [84], "dummy/4": [85],
                                "price/2": [812], "price/3": [813], "price/4": [814]}),
             arr=pd.DataFrame({"dummy": [83, 84, 85], "price": [812, 813, 814]})),
    Sequence(price=15, tradeable=True, available=False, date=pd.Period("01-18-2021"), sentiment_data_available=True,
             df=None, price_raw=150,
             flat=pd.DataFrame({"dummy/2": [84], "dummy/3": [85], "dummy/4": [86],
                                "price/2": [813], "price/3": [814], "price/4": [815]}),
             arr=pd.DataFrame({"dummy": [84, 85, 86], "price": [813, 814, 816]}))
]

for seq in t1.sequences:
    seq.add_eval(1, {"hold": 0.3, "buy": 0.4, "sell": 0.3})

for seq in t2.sequences:
    seq.add_eval(1, {"hold": 0.3, "buy": 0.4, "sell": 0.3})

data = [t1, t2]


def test_rename_actions():
    ev = Evaluate(ticker=copy.deepcopy(data))
    ev._rename_actions()

    result = ev.ticker

    for r in result:
        for seq in r.sequences:
            assert seq.action in ["hold", "buy", "sell"]


def test_min_max_date():
    ev = Evaluate(ticker=copy.deepcopy(data))
    ev._merge_sequence_attributes_to_df()
    ev._find_min_max_date()

    min_, max_ = ev._min_date, ev._max_date
    assert min_ == pd.Period("01-01-2021") and max_ == pd.Period("01-18-2021")


def test_merge_sequence_attributes_to_df():
    ev = Evaluate(ticker=copy.deepcopy(data))
    ev._merge_sequence_attributes_to_df()

    result = ev._sequence_attributes_df
    expected = pd.DataFrame({'hold': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                             'buy': [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                             'sell': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                             'dates': [pd.Period('2021-01-01', 'D'), pd.Period('2021-01-02', 'D'),
                                       pd.Period('2021-01-03', 'D'), pd.Period('2021-01-15', 'D'),
                                       pd.Period('2021-01-16', 'D'), pd.Period('2021-01-17', 'D'),
                                       pd.Period('2021-01-18', 'D')],
                             'actions': [1] * 7})

    assert_frame_equal(result.reset_index(drop=True), expected, check_dtype=False)


def test_set_quantile_thresholds():
    ev = Evaluate(ticker=copy.deepcopy(data))
    ev._rename_actions()
    ev._merge_sequence_attributes_to_df()

    quantiles = {"hold": 0.5, "buy": 0.5}

    with pytest.raises(AssertionError):
        ev.set_quantile_thresholds(quantiles)

    quantiles = {"hold": 0.5, "buy": 0.5, "sell": None}
    ev.set_quantile_thresholds(quantiles)
    result = ev.thresholds
    assert result["sell"] == 0 and result["hold"] == 0 and result["buy"] == 0.4


def test_get_dates_trades_combination():
    ev = Evaluate(ticker=copy.deepcopy(data))
    ev._rename_actions()
    ev._merge_sequence_attributes_to_df()
    ev._find_min_max_date()
    ev._get_dates_trades_combination()

    result = ev._dates_trades_combination

    assert (result["01-01-2021"][0].price == 120 and
            result["17-01-2021"][0].price == 140 and
            result["18-01-2021"][0].price == 150)


def test_act_1():
    actions = [1, 1, 2, 2]
    thresholds = [{"hold": 0.3, "buy": 0.4, "sell": 0.3}] * 4

    i = 0
    while i < len(t1.sequences):
        t1.sequences[i].add_eval(actions[i], thresholds[i])
        i += 1

    data = [t1]

    ev = Evaluate(ticker=copy.deepcopy(data), max_price_per_stock=1000, initial_balance=10000,
                  max_investment_per_trade=0.5, partial_shares_possible=False)
    ev.initialize()
    ev.set_thresholds({"hold": 0, "buy": 0, "sell": 0})
    ev.act()

    assert ev.balance == 10910.53


def test_act_2():
    actions = [1, 1, 2, 2]
    thresholds = [{"hold": 0.3, "buy": 0.4, "sell": 0.3}] * 4

    i = 0
    while i < len(t2.sequences):
        t2.sequences[i].add_eval(actions[i], thresholds[i])
        i += 1
    data = [t2]
    ev = Evaluate(ticker=copy.deepcopy(data), max_price_per_stock=1000, initial_balance=10000,
                  max_investment_per_trade=0.5, partial_shares_possible=False)
    ev.initialize()
    ev.set_thresholds({"hold": 0, "buy": 0, "sell": 0})
    ev.act()

    assert ev.balance == 10910.53


def test_act_threshold():
    # Shall only buy the second buy action since the firsts threshold is too low
    actions = [1, 1, 2]
    thresholds = [{"hold": 0.3, "buy": 0.4, "sell": 0.3}, {"hold": 0.3, "buy": 0.5, "sell": 0.3},
                  {"hold": 0.3, "buy": 0.4, "sell": 0.3}]

    i = 0
    while i < len(t1.sequences):
        t1.sequences[i].add_eval(actions[i], thresholds[i])
        i += 1

    data = [t1]

    ev = Evaluate(ticker=copy.deepcopy(data), max_price_per_stock=1000, initial_balance=10000,
                  max_investment_per_trade=0.5, partial_shares_possible=False)
    ev.initialize()
    ev.set_thresholds({"hold": 0, "buy": 0.5, "sell": 0})
    ev.act()
    assert round(ev.balance, 3) == 10240.130


def test_force_sell():
    # TBD
    pass
