class PreEnv:

    def __init__(self, ticker):
        self.ticker = ticker

    def exclude_non_tradeable_sequences(self):
        new_ticker = []
        for tck in self.ticker:
            new_sequences = [seq for seq in tck.sequences if seq.tradeable is True]
            if len(new_sequences) != 0:
                tck.sequences = new_sequences
                new_ticker.append(tck)
        self.ticker = new_ticker

    def get_updated_ticker(self):
        return self.ticker
