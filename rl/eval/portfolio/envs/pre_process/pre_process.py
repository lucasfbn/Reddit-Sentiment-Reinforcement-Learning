class Pre:

    def __init__(self):
        pass

    def run(self, ticker):
        raise NotImplementedError


class PreProcessor(Pre):

    def run(self, ticker):
        return ticker
