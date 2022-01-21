from math import e


class RewardHandler:
    COMPLETED_STEPS_MAX_REWARD = 3
    FORCED_EPISODE_END_PENALTY = 25
    TOTAL_EPISODE_END_REWARD = 5
    FLAT_REWARD = 0.05
    I_DISCOUNT = 0.0065

    def negate_if_no_success(self, reward, success):
        if not success:
            reward *= -1 if reward > 0 else 2
        return reward

    def discount_cash_bound(self, reward, cash_bound):
        if reward > 0:
            reward = reward / (1 + self.I_DISCOUNT) ** cash_bound  # present value formula
        return reward

    def add_flat_reward(self, reward):
        reward += self.FLAT_REWARD
        return reward

    def add_reward_completed_steps(self, reward, completed_steps_perc):
        reward += self.COMPLETED_STEPS_MAX_REWARD * completed_steps_perc
        return reward

    def discount_n_trades_left(self, reward, n_trades_left_perc):
        reward *= n_trades_left_perc
        return reward

    def penalize_forced_episode_end(self, reward, forced_episode_end):
        if forced_episode_end:
            reward -= self.FORCED_EPISODE_END_PENALTY
        return reward

    def reward_total_episode_end(self, reward, total_episode_end):
        if total_episode_end:
            reward += self.TOTAL_EPISODE_END_REWARD
        return reward
