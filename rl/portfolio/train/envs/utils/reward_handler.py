class RewardHandler:
    I_DISCOUNT = 0.065
    PENALTY_BASE = -3

    def discount_cash_bound(self, reward, cash_bound):
        if reward > 0:
            reward = reward / (1 + self.I_DISCOUNT) ** cash_bound  # present value formula
        return reward

    @staticmethod
    def _min_max_scaler(min_, max_, x, a, b):
        return a + (((x - min_) * (b - a)) / (max_ - min_))

    def penalize_func(self, x):
        if x <= 0.2:
            return 1 - (x / 0.3) ** 0.4
        elif 0.2 < x <= 0.25:
            return 0
        elif 0.25 < x <= 1.0:
            x = self._min_max_scaler(0.25, 1.0, x, 0, 1)
            return x ** 1.4

    def penalize_ratio(self, reward, trades_ratio):
        return reward + (self.PENALTY_BASE * self.penalize_func(trades_ratio))
