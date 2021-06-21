class Ticker:

    def __init__(self, name, df):
        self.df = df
        self.name = name
        self.exclude = False

        self.sequences = None

        self.actions = None

    def add_eval(self, actions):
        self.actions = actions
