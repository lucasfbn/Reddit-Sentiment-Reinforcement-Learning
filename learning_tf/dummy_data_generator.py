import numpy as np


class DummyData:

    def __init__(self, n, break_every=10):
        self.n = n
        self.break_every = break_every

    def generate(self):

        data = []
        break_data = []

        for i in range(self.n):

            arr = np.array([i])
            arr = arr.reshape((1, 1))
            break_data.append(arr)

            if i % self.break_every == 0:
                data.append({"data": break_data})
                break_data = []

        return data


dd = DummyData(1000).generate()
