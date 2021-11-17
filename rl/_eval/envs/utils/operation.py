class Operation:

    def __init__(self, ticker, sequence):
        self.action_probas = sequence.action_probas
        self.action = sequence.action
        self.tradeable = sequence.tradeable
        self.date = sequence.date
        self.price = sequence.price_raw
        self.ticker = ticker.name
        self.sequence = sequence
