import pickle as pkl
import random

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from preprocessing.sequences import Sequence
from preprocessing.tasks import Ticker
from rl.env import Env
from rl.env import EnvCNN, EnvNN

ticker = [Ticker(_, None) for _ in range(5)]
for tck in ticker:
    tck.sequences = [tck.name for _ in range(5)]

t1 = Ticker(None, None)
t2 = Ticker(None, None)

t1.sequences = [
    Sequence(price=12, tradeable=True, available=False, date=None, sentiment_data_available=None,
             df=None,
             flat=pd.DataFrame({"dummy/0": [1], "dummy/1": [2], "dummy/2": [3],
                                "price/0": [10], "price/1": [11], "price/2": [12]}),
             arr=pd.DataFrame({"dummy": [1, 2, 3], "price": [10, 11, 12]}), price_raw=120),
    Sequence(price=13, tradeable=True, available=False, date=None, sentiment_data_available=None,
             df=None,
             flat=pd.DataFrame({"dummy/1": [2], "dummy/2": [3], "dummy/3": [4],
                                "price/1": [11], "price/2": [12], "price/3": [13]}),
             arr=pd.DataFrame({"dummy": [2, 3, 4], "price": [11, 12, 13]}), price_raw=130),
    Sequence(price=14, tradeable=True, available=False, date=None, sentiment_data_available=None,
             df=None,
             flat=pd.DataFrame({"dummy/2": [3], "dummy/3": [4], "dummy/4": [5],
                                "price/2": [12], "price/3": [13], "price/4": [14]}),
             arr=pd.DataFrame({"dummy": [3, 4, 5], "price": [12, 13, 14]}), price_raw=140)]

t2.sequences = [
    Sequence(price=12, tradeable=True, available=False, date=None, sentiment_data_available=None,
             df=None,
             flat=pd.DataFrame({"dummy/0": [81], "dummy/1": [82], "dummy/2": [83],
                                "price/0": [810], "price/1": [811], "price/2": [812]}),
             arr=pd.DataFrame({"dummy": [81, 82, 83], "price": [810, 811, 812]}), price_raw=121),
    Sequence(price=13, tradeable=True, available=False, date=None, sentiment_data_available=None,
             df=None,
             flat=pd.DataFrame({"dummy/1": [82], "dummy/2": [83], "dummy/3": [84],
                                "price/1": [811], "price/2": [812], "price/3": [813]}),
             arr=pd.DataFrame({"dummy": [82, 83, 84], "price": [811, 812, 813]}), price_raw=131),
    Sequence(price=14, tradeable=True, available=False, date=None, sentiment_data_available=None,
             df=None,
             flat=pd.DataFrame({"dummy/2": [83], "dummy/3": [84], "dummy/4": [85],
                                "price/2": [812], "price/3": [813], "price/4": [814]}),
             arr=pd.DataFrame({"dummy": [83, 84, 85], "price": [812, 813, 814]}), price_raw=141)]

data = [t1, t2]


def test_basic_env_cnn_run():
    env = EnvCNN(ticker=data)

    expected_states = [
        np.array([[1, 2, 3], [10, 11, 12]], dtype=np.float32).T,
        np.array([[2, 3, 4], [11, 12, 13]], dtype=np.float32).T,
        np.array([[3, 4, 5], [12, 13, 14]], dtype=np.float32).T,
        np.array([[81, 82, 83], [810, 811, 812]], dtype=np.float32).T,
        np.array([[82, 83, 84], [811, 812, 813]], dtype=np.float32).T,
        np.array([[83, 84, 85], [812, 813, 814]], dtype=np.float32).T
    ]

    i = 0
    for _ in range(2):
        state = env.reset()
        terminal = False

        while not terminal:
            assert_array_equal(state, expected_states[i].reshape(1, 3, 2))
            actions = random.randint(0, 2)
            state, terminal, reward = env.execute(actions=actions)

            i += 1

    assert i == len(expected_states)


def test_basic_env_nn_run():
    env = EnvNN(data)

    expected_states = [
        np.array([[1, 2, 3, 10, 11, 12]], dtype=np.float32),
        np.array([[2, 3, 4, 11, 12, 13]], dtype=np.float32),
        np.array([[3, 4, 5, 12, 13, 14]], dtype=np.float32),
        np.array([[81, 82, 83, 810, 811, 812]], dtype=np.float32),
        np.array([[82, 83, 84, 811, 812, 813]], dtype=np.float32),
        np.array([[83, 84, 85, 812, 813, 814]], dtype=np.float32)
    ]

    i = 0
    for _ in range(2):
        state = env.reset()
        terminal = False

        while not terminal:
            assert_array_equal(state, expected_states[i].reshape(6, ))
            actions = random.randint(0, 2)
            state, terminal, reward = env.execute(actions=actions)

            i += 1

    assert i == len(expected_states)


def test_reward():
    env = EnvCNN(data)

    expected_rewards = [-data[0].sequences[0].price * env.TRANSACTION_COSTS_PERC,
                        -data[0].sequences[1].price * env.TRANSACTION_COSTS_PERC,
                        3, 0, -data[1].sequences[1].price * env.TRANSACTION_COSTS_PERC, 1]
    actions = [1, 1, 2, 0, 1, 2]
    i = 0
    for _ in range(2):
        state = env.reset()
        terminal = False

        while not terminal:
            state, terminal, reward = env.execute(actions=actions[i])
            assert reward == expected_rewards[i]
            i += 1


def test_w_real_data_cnn():
    with open("ticker.pkl", "rb") as f:
        data = pkl.load(f)

    env = EnvCNN(data)

    for _ in range(100):
        states = env.reset()
        terminal = False

        while not terminal:
            actions = random.randint(0, 2)
            states, terminal, reward = env.execute(actions=actions)


def test_w_real_data_nn():
    with open("ticker.pkl", "rb") as f:
        data = pkl.load(f)

    env = EnvNN(data)

    for _ in range(100):
        states = env.reset()
        terminal = False

        while not terminal:
            actions = random.randint(0, 2)
            states, terminal, reward = env.execute(actions=actions)


def test_cnn_loop():
    """
    Should loop through the whole data twice
    """
    env = EnvCNN(data)

    expected_states = [
        np.array([[1, 2, 3], [10, 11, 12]], dtype=np.float32).T,
        np.array([[2, 3, 4], [11, 12, 13]], dtype=np.float32).T,
        np.array([[3, 4, 5], [12, 13, 14]], dtype=np.float32).T,
        np.array([[81, 82, 83], [810, 811, 812]], dtype=np.float32).T,
        np.array([[82, 83, 84], [811, 812, 813]], dtype=np.float32).T,
        np.array([[83, 84, 85], [812, 813, 814]], dtype=np.float32).T
    ]

    i = 0
    for _ in range(4):
        state = env.reset()
        terminal = False

        while not terminal:
            assert_array_equal(state, expected_states[i].reshape(1, 3, 2))
            actions = random.randint(0, 2)
            state, terminal, reward = env.execute(actions=actions)

            i += 1

        if i == len(expected_states):
            i = 0


def test_max_episode_timesteps():
    env = EnvNN(data)
    assert env.max_episode_timesteps() == 3


def test_nn_states():
    env = EnvNN(data)
    assert env.states() == {'shape': (6,), 'type': 'float'}


def test_cnn_states():
    env = EnvCNN(data)
    assert env.states() == {'shape': (1, 3, 2), 'type': 'float'}


# def test_exclude_non_tradeable_sequences():
#     t2.sequences[2].tradeable = False
#
#     EnvCNN.data = data
#     EnvCNN.shuffle_sequences = False
#
#     env = EnvCNN()
#
#     expected_states = [
#         np.array([[1, 2, 3], [10, 11, 12]], dtype=np.float32).T,
#         np.array([[2, 3, 4], [11, 12, 13]], dtype=np.float32).T,
#         np.array([[3, 4, 5], [12, 13, 14]], dtype=np.float32).T,
#         np.array([[81, 82, 83], [810, 811, 812]], dtype=np.float32).T,
#         np.array([[82, 83, 84], [811, 812, 813]], dtype=np.float32).T,
#     ]
#
#     i = 0
#     for _ in range(2):
#         state = env.reset()
#         terminal = False
#
#         while not terminal:
#             assert_array_equal(state, expected_states[i].reshape(1, 3, 2))
#             actions = random.randint(0, 2)
#             state, terminal, reward = env.execute(actions=actions)
#
#             i += 1
#
#     assert i == len(expected_states)


def test_next_ticker():
    e = Env(ticker=ticker)

    result_curr_ticker = []
    result_curr_seq = []
    expected_curr_ticker = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]

    for i in range(12):
        e.next_ticker()
        result_curr_ticker.append(e.curr_ticker)
        result_curr_seq.append(e.curr_sequences[0])

    result_curr_ticker = [tck.name for tck in result_curr_ticker]

    assert result_curr_ticker == expected_curr_ticker
    assert result_curr_seq == expected_curr_ticker
