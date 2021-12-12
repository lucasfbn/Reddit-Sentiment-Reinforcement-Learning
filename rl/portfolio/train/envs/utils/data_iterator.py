class DataIterator:

    def __init__(self, sequences):
        self._sequences = sequences
        self._index = 0

        self.episode_end = False
        self.episode_count = 0
        self.curr_sequence = None

    def next_sequence(self):
        self.curr_sequence = self._sequences[self._index]
        self._index += 1

        if self._index == len(self._sequences):
            self.episode_end = True
            self._index = 0

        return self.curr_sequence


di = DataIterator([1, 2, 3, 4, 5])
di.next_sequence()

for _ in range(15):
    print(di.curr_sequence, di.episode_end)

    if di.episode_end is True:
        di.episode_end = False

    di.next_sequence()
