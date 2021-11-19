import pytest

from rl._eval.envs.sub_envs.trading import TradingSimulator
from tests.utils import MockObj


def new_env():
    env = TradingSimulator()
    env.START_BALANCE = 1000
    env.INVESTMENT_PER_TRADE = 70
    env.MAX_PRICE_PER_STOCK = 10
    env.MAX_TRADES_PER_DAY = 3
    env.SLIPPAGE = 0.007
    env.ORDER_FEE = 0.02
    return env


def test_buy():
    env = new_env()

    operation = MockObj(tradeable=True, price=10, ticker="1")

    success = env.buy(operation)
    assert success is True
    assert len(env._inventory) == 1
    assert env._balance == 938.38
    assert env._n_trades == 1

    env.reset_day()

    assert env._n_trades == 0


def test_sell():
    env = new_env()
    operation1 = MockObj(tradeable=True, price=10, ticker="1")
    _ = env.buy(operation1)

    # No sell because no stock in inventory
    operation2 = MockObj(tradeable=True, price=20, ticker="2")
    success = env.sell(operation2)

    assert success is True
    assert len(env._inventory) == 1

    # Sell
    operation2 = MockObj(tradeable=True, price=20, ticker="1")
    success = env.sell(operation2)
    assert success is True
    assert len(env._inventory) == 0
    assert env._balance == 1055.14


def test_buy_checks():
    env = new_env()

    operation = MockObj(tradeable=False, price=10, ticker="1")
    assert env.buy(operation) is False

    operation = MockObj(tradeable=True, price=10, ticker="1")
    env._balance = 15
    assert env.buy(operation) is False

    operation = MockObj(tradeable=True, price=10, ticker="1")
    env._n_trades = 2
    assert env.buy(operation) is False

    operation = MockObj(tradeable=True, price=10, ticker="1")
    env.MAX_PRICE_PER_STOCK = 5
    assert env.buy(operation) is False
