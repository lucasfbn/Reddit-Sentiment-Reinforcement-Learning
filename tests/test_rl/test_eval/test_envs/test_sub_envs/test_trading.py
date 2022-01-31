from dataset_handler.classes.sequence import Sequence
from rl.simulation.envs.sub_envs.trading import TradingSimulator


def test_sequence(tradeable, price, ticker):
    seq = Sequence()
    seq.metadata.ticker_name = ticker
    seq.metadata.price_raw = price
    seq.metadata.tradeable = tradeable
    return seq


def new_env():
    TradingSimulator.START_BALANCE = 1000
    TradingSimulator.INVESTMENT_PER_TRADE = 70
    TradingSimulator.MAX_PRICE_PER_STOCK = 10
    TradingSimulator.SLIPPAGE = 0.007
    TradingSimulator.ORDER_FEE = 0.02
    return TradingSimulator()


def test_buy():
    env = new_env()

    seq = test_sequence(True, 10, "1")

    success = env.buy(seq)
    assert success is True
    assert len(env._inventory) == 1
    assert env._balance == 938.38
    assert env._n_trades == 1

    env.reset_day()

    assert env._n_trades == 0


def test_sell():
    env = new_env()
    operation1 = test_sequence(tradeable=True, price=10, ticker="1")
    _ = env.buy(operation1)

    # No sell because no stocks in inventory
    operation2 = test_sequence(tradeable=True, price=20, ticker="2")
    success = env.sell(operation2)

    assert success is True
    assert len(env._inventory) == 1

    # Sell
    operation2 = test_sequence(tradeable=True, price=20, ticker="1")
    success = env.sell(operation2)
    assert success is True
    assert len(env._inventory) == 0
    assert env._balance == 1055.14


def test_buy_checks():
    env = new_env()

    operation = test_sequence(tradeable=False, price=10, ticker="1")
    assert env.buy(operation) is False

    operation = test_sequence(tradeable=True, price=10, ticker="1")
    env._balance = 15
    assert env.buy(operation) is False

    operation = test_sequence(tradeable=True, price=10, ticker="1")
    env._n_trades = 2
    assert env.buy(operation) is False

    operation = test_sequence(tradeable=True, price=10, ticker="1")
    env.MAX_PRICE_PER_STOCK = 5
    assert env.buy(operation) is False


def test_inventory_state():
    env = new_env()
    operation = test_sequence(tradeable=True, price=10, ticker="1")
    env.buy(operation)

    assert env.inventory_state(operation) == 1

    env = new_env()
    operation = test_sequence(tradeable=True, price=10, ticker="1")
    assert env.inventory_state(operation) == 0
