class Operation:

    def __init__(self, ticker, price, date, tradeable, action, action_probas):
        self.action_probas = action_probas
        self.action = action
        self.tradeable = tradeable
        self.date = date
        self.price = price
        self.ticker = ticker

        self.price_bought = None
        self.quantity = None
        self.total_buy_price = None

    def save_buy(self, price, quantity, total_buy_price):
        self.price_bought = price
        self.quantity = quantity
        self.total_buy_price = total_buy_price
