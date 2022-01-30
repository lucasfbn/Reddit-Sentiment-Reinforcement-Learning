import numpy as np


class RewardHandler:
    COMPLETED_STEPS_MAX_REWARD = 1
    FORCED_EPISODE_END_PENALTY = 1
    TOTAL_EPISODE_END_REWARD = 25
    I_DISCOUNT = 0.0065

    def inv_trades_ratio(self, inv_len, n_trades):
        inv_ratio = inv_len / (inv_len + n_trades)
        trades_ratio = 1 - inv_ratio
        return inv_ratio, trades_ratio

    def discount_0_action_penatly(self, inv_len, n_trades):
        base = -.05
        inv_ratio, trades_ratio = self.inv_trades_ratio(inv_len, n_trades)
        return trades_ratio * base

    def scale_n_trades(self, start_n_trades, n_trades):
        max_multiplier = 6
        min_ = 0
        max_ = int(max_multiplier * start_n_trades)
        return (n_trades - min_) / (max_ - min_)

    def negate_if_no_success(self, reward, success):
        if not success:
            reward *= -1 if reward > 0 else 2
        return reward

    def discount_cash_bound(self, reward, cash_bound):
        if reward > 0:
            reward = reward / (1 + self.I_DISCOUNT) ** cash_bound  # present value formula
        return reward

    def add_reward_completed_steps(self, reward, completed_steps_perc):
        if reward > 0:
            reward * completed_steps_perc
        return reward

    def discount_n_trades_left(self, reward, n_trades_left_perc):
        if reward > 0:
            return (n_trades_left_perc * reward) ** 1.4
        return reward

    def penalize_forced_episode_end(self, reward, forced_episode_end):
        if forced_episode_end:
            reward -= self.FORCED_EPISODE_END_PENALTY
        return reward

    def reward_total_episode_end(self, reward, total_episode_end):
        if total_episode_end:
            reward += self.TOTAL_EPISODE_END_REWARD
        return reward

    def penalize_ratio(self, x):
        # https://dhemery.github.io/DHE-Modules/technical/sigmoid/#normalized
        k = 0.5
        x = self.min_max_scaler(0, 1, x, -1, 1)
        normalized_sigmoid = (x - k * x) / (k - 2 * k * abs(x) + 1)
        return abs(normalized_sigmoid)

    def min_max_scaler(self, min_, max_, x, a, b):
        return a + (((x - min_) * (b - a)) / (max_ - min_))

    def go(self, x):
        if x <= 0.3:
            # return 1 - (x / 0.1) ** 0.4
            # return -10 * x + 1
            x = self.min_max_scaler(0.3, 0.0, x, 0, 1)
            p0 = np.array([0.1, 0.0])
            p1 = np.array([0.0, 0.0])
            p2 = np.array([0.0, 1.0])
            bezier = (1 - x) ** 2 * p0 + 2 * (1 - x) * x * p1 + x ** 2 * p2
            return bezier[1]
        elif 0.3 < x <= 0.6:
            return 0.0
        elif 0.6 < x <= 1.0:
            x = self.min_max_scaler(0.6, 1.0, x, 0, 1)
            p0 = np.array([0.6, 0.0])
            p1 = np.array([0.9, 0.0])
            p2 = np.array([1.0, 0.0])
            p3 = np.array([1.0, 1.0])
            bezier = (-p0 + 3 * p1 - 3 * p2 + p3) * x ** 3 + (3 * p0 - 6 * p1 + 3 * p2) * x ** 2 + (
                    -3 * p0 + 3 * p1) * x + p0
            return bezier[1]
        else:
            raise ValueError
