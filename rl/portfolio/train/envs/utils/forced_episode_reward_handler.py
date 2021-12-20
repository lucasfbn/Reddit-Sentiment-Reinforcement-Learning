class ForcedEpisodeRewardHandler:
    INITIAL_REWARD_WHEN_FORCED_EPISODE_END = -25

    def __init__(self, start, end):
        self.start = start
        self.end = end

        self._possible_episodes = end - start
        self._discount_per_episode = self.INITIAL_REWARD_WHEN_FORCED_EPISODE_END / \
                                     self._possible_episodes

    def get_episode_end_reward(self, curr_episode):
        if curr_episode > self._possible_episodes:
            raise ValueError("Current episode greater than possible episodes.")

        return self._discount_per_episode * \
               (self._possible_episodes - curr_episode)

print(0/90)
