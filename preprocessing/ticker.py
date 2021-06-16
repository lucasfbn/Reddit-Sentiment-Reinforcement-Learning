class Ticker:

    def __init__(self, name, df):
        self.df = df
        self.name = name
        self.metadata = None
        self.exclude = False

        self.flat_sequence = None
        self.array_sequence = None