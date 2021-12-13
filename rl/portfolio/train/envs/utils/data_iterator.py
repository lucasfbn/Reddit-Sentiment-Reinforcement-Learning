class DataIterator:

    def __init__(self, sequences):
        self._sequences = sequences
        self._index = 0

        self.episode_end = False
        self.episode_count = 0

        self.last_sequence = None
        self.curr_sequence = None

    @property
    def sequences(self):
        return self._sequences

    def is_new_date(self):
        if self.last_sequence is None:
            return False
        return self.last_sequence.metadata.date != self.curr_sequence.metadata.date

    def next_sequence(self):
        self.last_sequence = self.curr_sequence
        self.curr_sequence = self._sequences[self._index]
        self._index += 1

        if self._index == len(self._sequences):
            self.episode_end = True
            self.last_sequence = None
            self._index = 0

        return self.curr_sequence
