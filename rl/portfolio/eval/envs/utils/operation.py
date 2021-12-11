class Operation:

    def __init__(self, ticker, sequence):
        self.tradeable = sequence.metadata.tradeable
        self.date = sequence.metadata.date
        self.price = sequence.metadata.price_raw
        self.ticker = ticker.name
        self.sequence = sequence
