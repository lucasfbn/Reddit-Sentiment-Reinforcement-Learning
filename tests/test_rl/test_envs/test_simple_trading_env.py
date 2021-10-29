from rl.envs.simple_trading import SimpleTradingEnv


#######################################################
### Most tests are already covered in ./test_env.py ###
#######################################################

def set_all_false(simple_trading_env_instance):
    simple_trading_env_instance.ENABLE_TRANSACTION_COSTS = False
    simple_trading_env_instance.ENABLE_NEG_BUY_REWARD = False
    simple_trading_env_instance.ENABLE_POS_SELL_REWARD = False
    return simple_trading_env_instance


def test_basic_buy():
    ste = SimpleTradingEnv("test")
    ste = set_all_false(ste)
    reward = ste.buy(10)

    assert reward == 0
    assert ste.inventory == [10]


def test_basic_buy_loop():
    ste = SimpleTradingEnv("test")
    ste = set_all_false(ste)

    prices = [10, 16, 20]

    for price in prices:
        reward = ste.buy(price)
        assert reward == 0

    assert ste.inventory == prices


def test_empty_sell():
    ste = SimpleTradingEnv("test")
    ste = set_all_false(ste)
    reward = ste.sell(10)

    assert reward == 0
    assert ste.inventory == []


def test_sell():
    ste = SimpleTradingEnv("test")
    ste = set_all_false(ste)

    _ = ste.buy(10)
    assert ste.inventory == [10]
    reward = ste.sell(15)

    assert reward == 5
    assert ste.inventory == []


def test_transaction_fee_ask():
    ste = SimpleTradingEnv("test")
    ste = set_all_false(ste)
    ste.ENABLE_TRANSACTION_COSTS = True
    ste.TRANSACTION_FEE_ASK = 0.005

    reward = ste.buy(10)
    assert reward == -(10 * 0.005)


def test_transaction_fee_bid():
    ste = SimpleTradingEnv("test")
    ste = set_all_false(ste)
    ste.ENABLE_TRANSACTION_COSTS = True
    ste.TRANSACTION_FEE_BID = 0.005

    ste.buy(10)
    reward = ste.sell(15)

    assert reward == (15 - 10) - 15 * 0.005


def test_neg_buy_reward():
    ste = SimpleTradingEnv("test")
    ste = set_all_false(ste)
    ste.ENABLE_NEG_BUY_REWARD = True

    reward = ste.buy(10)
    assert reward == -10


def test_pos_sell_reward():
    ste = SimpleTradingEnv("test")
    ste = set_all_false(ste)
    ste.ENABLE_POS_SELL_REWARD = True

    _ = ste.buy(10)
    reward = ste.sell(15)
    assert reward == 20


def test_neg_buy_reward_plus_transaction_fee():
    ste = SimpleTradingEnv("test")
    ste = set_all_false(ste)
    ste.ENABLE_NEG_BUY_REWARD = True
    ste.ENABLE_TRANSACTION_COSTS = True
    ste.TRANSACTION_FEE_ASK = 0.005

    reward = ste.buy(10)
    assert reward == -10 - (10 * 0.005)


def test_pos_sell_reward_plus_transaction_fee():
    ste = SimpleTradingEnv("test")
    ste = set_all_false(ste)
    ste.ENABLE_POS_SELL_REWARD = True
    ste.ENABLE_TRANSACTION_COSTS = True
    ste.TRANSACTION_FEE_BID = 0.005

    _ = ste.buy(10)
    reward = ste.sell(15)
    assert reward == 20 - (15 * 0.005)
