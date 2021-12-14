import pandas as pd

from rl.portfolio.train.envs.env import EnvCNNExtended
from rl.portfolio.train.envs.sub_envs.trading import TradingSimulator
from tests.utils import MockObj

data = [
    MockObj(evl=MockObj(buy_proba=0.9, reward_backtracked=1.0,
                        days_cash_bound=2),
            metadata=MockObj(ticker_name="A", date=1),
            data=MockObj(arr=pd.DataFrame({"dummy": [1]}))),
    MockObj(evl=MockObj(buy_proba=0.8, reward_backtracked=2.0,
                        days_cash_bound=2),
            metadata=MockObj(ticker_name="B", date=2),
            data=MockObj(arr=pd.DataFrame({"dummy": [1]}))),
    MockObj(evl=MockObj(buy_proba=0.7, reward_backtracked=3.0,
                        days_cash_bound=1),
            metadata=MockObj(ticker_name="C", date=2),
            data=MockObj(arr=pd.DataFrame({"dummy": [1]}))),
    MockObj(evl=MockObj(buy_proba=0.6, reward_backtracked=4.0,
                        days_cash_bound=1),
            metadata=MockObj(ticker_name="B", date=3),
            data=MockObj(arr=pd.DataFrame({"dummy": [1]}))),
    MockObj(evl=MockObj(buy_proba=0.5, reward_backtracked=5.0,
                        days_cash_bound=10),
            metadata=MockObj(ticker_name="C", date=4),
            data=MockObj(arr=pd.DataFrame({"dummy": [1]}))),
    MockObj(evl=MockObj(buy_proba=0.4, reward_backtracked=6.0,
                        days_cash_bound=10),
            metadata=MockObj(ticker_name="D", date=4),
            data=MockObj(arr=pd.DataFrame({"dummy": [1]})))
]


def test_reward_0_action():
    env = EnvCNNExtended(data)
    _ = env.reset()

    actions = [0, 0, 0, 0]

    for action in actions:
        next_state, reward, episode_end, _ = env.step(action)
        assert reward == 0.0


def test_reward_1_action():
    env = EnvCNNExtended(data)
    TradingSimulator.N_START_TRADES = 999
    _ = env.reset()

    actions = [1, 1, 1, 1]

    for i, action in enumerate(actions):
        next_state, reward, episode_end, _ = env.step(action)
        assert reward == data[i].evl.reward_backtracked


def test_reward_no_trades_left():
    env = EnvCNNExtended(data)
    TradingSimulator.N_START_TRADES = 0.9
    _ = env.reset()

    actions = [1, 1, 1, 1]

    for i, action in enumerate(actions):
        next_state, reward, episode_end, _ = env.step(action)
        assert reward == data[i].evl.reward_backtracked * -1

    TradingSimulator.N_START_TRADES = 20


def test_episode_ends():
    env = EnvCNNExtended(data)

    _ = env.reset()

    actions = [1] * 6
    expected_episode_end = [False] * 5 + [True]

    for i, action in enumerate(actions):
        next_state, reward, episode_end, _ = env.step(action)
        assert episode_end == expected_episode_end[i]


def test_inventory_states():
    test_data = [
        MockObj(evl=MockObj(buy_proba=0.9, reward_backtracked=1.0,
                            days_cash_bound=2),
                metadata=MockObj(ticker_name="A", date=1),
                data=MockObj(arr=pd.DataFrame({"dummy": [1]}))),
        MockObj(evl=MockObj(buy_proba=0.8, reward_backtracked=2.0,
                            days_cash_bound=2),
                metadata=MockObj(ticker_name="B", date=2),
                data=MockObj(arr=pd.DataFrame({"dummy": [1]}))),
        MockObj(evl=MockObj(buy_proba=0.6, reward_backtracked=4.0,
                            days_cash_bound=1),
                metadata=MockObj(ticker_name="B", date=3),
                data=MockObj(arr=pd.DataFrame({"dummy": [1]}))),
        MockObj(evl=MockObj(buy_proba=0.6, reward_backtracked=4.0,
                            days_cash_bound=1),
                metadata=MockObj(ticker_name="D", date=3),
                data=MockObj(arr=pd.DataFrame({"dummy": [1]}))),
    ]

    env = EnvCNNExtended(test_data)

    _ = env.reset()

    actions = [1] * 4

    A = MockObj(metadata=MockObj(ticker_name="A", date=1))
    B = MockObj(metadata=MockObj(ticker_name="B", date=1))

    expected_A = [1, 1, 0, 0]
    expected_B = [0, 1, 1, 1]

    for i, action in enumerate(actions):
        next_state, reward, episode_end, _ = env.step(action)
        inv = env._trading_env.inventory
        inv_state_A = inv.inventory_state(A)
        inv_state_B = inv.inventory_state(B)

        assert inv_state_A == expected_A[i]
        assert inv_state_B == expected_B[i]
