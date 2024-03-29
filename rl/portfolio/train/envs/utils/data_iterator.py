class DataIterator:

    def __init__(self, sequences):
        self._sequences = sequences

        self.steps = 0

    @property
    def sequences(self):
        return self._sequences

    @property
    def perc_completed_steps(self):
        return self.steps / len(self._sequences)

    def step(self):
        self.steps += 1

    def _is_episode_end(self, index):
        return index == len(self._sequences) - 1

    @staticmethod
    def _is_new_date(last, curr):
        if last is None:
            return False
        return last.metadata.date != curr.metadata.date

    def sequence_iter(self):
        while True:
            last = None
            for i, seq in enumerate(self._sequences):
                yield seq, self._is_episode_end(i), self._is_new_date(last, seq)
                last = seq
