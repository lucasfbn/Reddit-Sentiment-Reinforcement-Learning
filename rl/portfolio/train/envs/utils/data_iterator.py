class DataIterator:

    def __init__(self, sequences):
        self._sequences = sequences
        self._index = 0

        self.episode_end = False
        self.episode_count = 0
        self.new_date = False

        self.curr_sequence = None

    def next_sequence(self):
        next_sequence = self._sequences[self._index]
        self._index += 1

        new_date = False
        if self.curr_sequence is not None:
            last_date = self.curr_sequence.metadata.date
            new_date = last_date != next_sequence.metadata.date

        if self._index == len(self._sequences):
            self.episode_end = True
            self._index = 0

        self.curr_sequence = next_sequence
        return self.curr_sequence, new_date
