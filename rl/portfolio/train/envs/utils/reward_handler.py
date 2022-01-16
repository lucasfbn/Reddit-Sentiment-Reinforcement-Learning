from math import e


class RewardHandler:
    COMPLETED_STEPS_MAX_REWARD = 25
    FORCED_EPISODE_END_PENALTY = 25
    TOTAL_EPISODE_END_REWARD = 25

    def __init__(self, base_reward):
        self.reward = base_reward

    def negate_if_no_success(self, success):
        if not success:
            self.reward *= -1 if self.reward > 0 else 2
        return self.reward

    def discount_cash_bound(self, cash_bound):
        if self.reward > 0:
            self.reward = self.reward * e ** -(0.03 * cash_bound)
        return self.reward

    def add_reward_completed_steps(self, completed_steps_perc):
        self.reward += self.COMPLETED_STEPS_MAX_REWARD * completed_steps_perc
        return self.reward

    def discount_n_trades_left(self, n_trades_left_perc):
        self.reward *= n_trades_left_perc
        return self.reward

    def penalize_forced_episode_end(self, forced_episode_end):
        if forced_episode_end:
            self.reward -= self.FORCED_EPISODE_END_PENALTY
        return self.reward

    def reward_total_episode_end(self, total_episode_end):
        if total_episode_end:
            self.reward += self.TOTAL_EPISODE_END_REWARD
        return self.reward
