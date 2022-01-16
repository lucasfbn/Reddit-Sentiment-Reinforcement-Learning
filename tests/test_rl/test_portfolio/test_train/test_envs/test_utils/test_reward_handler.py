from rl.portfolio.train.envs.utils.reward_handler import RewardHandler


def get_class():
    rh = RewardHandler(10)
    rh.COMPLETED_STEPS_MAX_REWARD = 25
    rh.FORCED_EPISODE_END_PENALTY = 25
    rh.TOTAL_EPISODE_END_REWARD = 25
    return rh


def test_negate_if_no_success():
    rh = get_class()
    assert rh.negate_if_no_success(False) == -10

    rh = get_class()
    assert rh.negate_if_no_success(True) == 10

    rh = get_class()
    rh.reward = -10
    assert rh.negate_if_no_success(False) == -20


def test_discount_cash_bound():
    rh = get_class()
    rh.reward = -10
    assert rh.discount_cash_bound(100) == -10

    rh = get_class()
    assert rh.discount_cash_bound(100) == 0.4978706836786395


def test_add_reward_completed_steps():
    rh = get_class()
    assert rh.add_reward_completed_steps(0.5) == 12.5 + 10

    rh = get_class()
    assert rh.add_reward_completed_steps(0.25) == 6.25 + 10


def test_discount_n_trades_left():
    rh = get_class()
    assert rh.discount_n_trades_left(0.5) == 5


def test_penalize_forced_episode_end():
    rh = get_class()
    assert rh.penalize_forced_episode_end(True) == -15

    rh = get_class()
    assert rh.penalize_forced_episode_end(False) == 10


def test_reward_total_episode_end():
    rh = get_class()
    assert rh.reward_total_episode_end(True) == 35

    rh = get_class()
    assert rh.reward_total_episode_end(False) == 10
