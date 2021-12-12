from rl.portfolio.train.envs.sub_envs.trading import TradingSimulator, Inventory
from tests.utils import MockObj

sequences = [
    MockObj(evl=MockObj(days_cash_bound=2,
                        reward_backtracked=0.5),
            metadata=MockObj(name="A")),
    MockObj(evl=MockObj(days_cash_bound=2,
                        reward_backtracked=-0.1),
            metadata=MockObj(name="B")),
]


def test_inventory_add():
    inv = Inventory()
    inv.add(sequences[0])

    assert inv._inv[0]["removal_day"] == 2


def test_inventory_state():
    inv = Inventory()
    assert inv.inventory_state(sequences[0]) == 0

    inv.add(sequences[0])
    assert inv.inventory_state(sequences[0]) == 1


def test_inventory_new_day():
    inv = Inventory()
    inv.add(sequences[0])
    inv.add(sequences[1])

    total_reward = inv.new_day()
    assert total_reward == 0
    assert len(inv._inv) == 2

    assert inv.new_day() == 2.4


def test_step_0_action():
    ts = TradingSimulator()
    assert ts.step(0, sequences[0]) == 0


def test_step_1_action():
    ts = TradingSimulator()
    assert ts.step(1, sequences[0]) == 0.5


def test_no_trades_left():
    ts = TradingSimulator()
    ts._n_trades = 1
    ts.step(1, sequences[0])

    assert ts.step(1, sequences[0]) == -0.5
    assert ts.step(1, sequences[1]) == -0.2

    ts.new_day()
    ts.new_day()
    assert ts._n_trades == 1.5
    assert ts.step(1, sequences[0]) == 0.5
