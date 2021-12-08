class Ticker:

    def __init__(self, name, df):
        self.df = df
        self.name = name
        self.exclude = False

        self.sequences = None

    def __len__(self):
        return len(self.sequences)
