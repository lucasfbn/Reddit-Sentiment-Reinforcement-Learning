import logging

logger = logging.getLogger()


class SimpleTradingEnv:
    TRANSACTION_FEE_ASK = 0.005  # Buy
    TRANSACTION_FEE_BID = 0.01  # Sell

    ENABLE_TRANSACTION_COSTS = False
    ENABLE_NEG_BUY_REWARD = False
    ENABLE_POS_SELL_REWARD = False

    def __init__(self, ticker_name):
        self.ticker_name = ticker_name
        self.inventory = []

    def hold_callback(self, reward, price):
        return reward

    def sell_callback(self, reward, price):
        return reward

    def buy_callback(self, reward, price):
        return reward

    def calculate_margin(self, current_price):
        margin = 0

        for buy_price in self.inventory:
            margin += current_price - buy_price

        return margin

    def hold(self, price):
        reward = 0

        reward = self.hold_callback(reward, price)

        return reward

    def buy(self, price):
        reward = 0
        self.inventory.append(price)

        if self.ENABLE_NEG_BUY_REWARD:
            reward -= price

        if self.ENABLE_TRANSACTION_COSTS:
            reward -= price * self.TRANSACTION_FEE_ASK

        logger.debug(f"BUY. Stock: {self.ticker_name}. Relative price: {price}")
        return reward

    def sell(self, price):
        reward = 0

        if len(self.inventory) > 0:

            margin = self.calculate_margin(price)

            if self.ENABLE_TRANSACTION_COSTS:
                reward -= price * self.TRANSACTION_FEE_BID

            if self.ENABLE_POS_SELL_REWARD:
                reward += price

            reward += margin

            self.inventory = []

            logger.debug(f"SELL. Stock: {self.ticker_name}. Relative price: {price}. Margin: {margin}")
        else:
            logger.debug(f"ATTEMPTED SELL. Stock: {self.ticker_name}. Inventory is empty.")

        return reward


class SimpleTradingEnvEvaluation(SimpleTradingEnv):
    TRANSACTION_FEE_ASK = 0.005  # Buy
    TRANSACTION_FEE_BID = 0.01  # Sell

    ENABLE_TRANSACTION_COSTS = True
    ENABLE_NEG_BUY_REWARD = True
    ENABLE_POS_SELL_REWARD = True


class SimpleTradingEnvTraining(SimpleTradingEnv):
    PARTIAL_HOLD_REWARD = False
    HOLD_REWARD_MULTIPLIER = 0.1

    def hold_callback(self, reward, price):
        if not self.PARTIAL_HOLD_REWARD:
            return reward

        margin = self.calculate_margin(price)
        reward += margin * self.HOLD_REWARD_MULTIPLIER
        return reward

    def sell_callback(self, reward, price):
        return reward

    def buy_callback(self, reward, price):
        return reward
