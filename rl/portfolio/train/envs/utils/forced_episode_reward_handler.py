class ForcedEpisodeRewardHandler:
    INITIAL_REWARD_WHEN_FORCED_EPISODE_END = -25

    def __init__(self, max_episodes):
        self.max_episodes = max_episodes

        self._discount_per_episode = self.INITIAL_REWARD_WHEN_FORCED_EPISODE_END / \
                                     self.max_episodes

    def get_episode_end_reward(self, curr_episode):
        if curr_episode > self.max_episodes:
            raise ValueError(f"Current episode ({curr_episode}) greater than"
                             f" possible episodes ({self.max_episodes}).")

        return self._discount_per_episode * \
               (self.max_episodes - curr_episode)
