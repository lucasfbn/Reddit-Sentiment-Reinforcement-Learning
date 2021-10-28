import logging

logger = logging.getLogger()


class SimpleTradingEnv:
    ENABLE_TRANSACTION_COSTS = False

    TRANSACTION_FEE_ASK = 0.005  # Buy
    TRANSACTION_FEE_BID = 0.01  # Sell

    ENABLE_NEG_BUY_REWARD = False

    def __init__(self, ticker_name):
        self.ticker_name = ticker_name
        self.inventory = []
        self.profit = 1

    def calculate_margin(self, current_price):
        margin = 0

        for buy_price in self.inventory:
            margin += current_price - buy_price

        return margin

    def hold(self, reward, price):
        return reward

    def buy(self, reward, price):
        self.inventory.append(price)

        if self.ENABLE_NEG_BUY_REWARD:
            reward -= price

        if self.ENABLE_TRANSACTION_COSTS:
            reward -= price * self.TRANSACTION_FEE_ASK

        logger.debug(f"BUY. Stock: {self.ticker_name}. Relative price: {price}")
        return reward

    def sell(self, reward, price):
        if len(self.inventory) > 0:

            margin = self.calculate_margin(price)

            if self.ENABLE_TRANSACTION_COSTS:
                margin -= margin * self.TRANSACTION_FEE_BID

            reward += margin

            self.inventory = []
            self.profit += margin

            logger.debug(f"SELL. Stock: {self.ticker_name}. Relative price: {price}. Margin: {margin}")
        else:
            logger.debug(f"ATTEMPTED SELL. Stock: {self.ticker_name}. Inventory is empty.")

        return reward
