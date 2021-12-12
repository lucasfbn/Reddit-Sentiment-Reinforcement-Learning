from collections import deque


class DataIterator:

    def __init__(self, ticker: list):
        self.ticker = ticker
        self.ticker_iter = 0
        self.ticker_iter_max = len(self.ticker)

        self.episode_end = False
        self.episode_count = 0

        self.curr_ticker = None
        self.curr_sequences = None
        self.curr_sequence = None

    def next_ticker(self):
        if self.ticker_iter == self.ticker_iter_max:
            self.ticker_iter = 0

        r = self.ticker[self.ticker_iter]

        self.ticker_iter += 1
        self.curr_ticker = r
        self.curr_sequences = deque(self.curr_ticker.sequences)

    def next_sequence(self):
        if len(self.curr_sequences) == 0:
            self.episode_end = True
        else:
            next_sequence = self.curr_sequences.popleft()
            self.curr_sequence = next_sequence
        return self.curr_sequence

    def is_episode_end(self):
        return self.episode_end
