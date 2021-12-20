import pytest

from rl.portfolio.train.envs.utils.forced_episode_reward_handler import ForcedEpisodeRewardHandler


def test_base():
    ForcedEpisodeRewardHandler.INITIAL_REWARD_WHEN_FORCED_EPISODE_END = -25.0
    e = ForcedEpisodeRewardHandler(90)
    assert e.get_episode_end_reward(0) == -25.0
    assert e.get_episode_end_reward(90) == -0.0
    assert e.get_episode_end_reward(45) == -12.5

    with pytest.raises(ValueError):
        assert e.get_episode_end_reward(91)
