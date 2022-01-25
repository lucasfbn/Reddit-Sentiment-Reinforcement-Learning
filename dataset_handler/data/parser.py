import h5py
import pandas as pd


class Parser:

    def __init__(self, fp):
        self.fp = fp
        self.store = h5py.File(self.fp, "r")

    def parse_single(self, obj):
        name = obj.name

        cols = self.store[f"{name}/COLS"][:].astype(str)
        arr = self.store[f"{name}/DF"][:]

        obj.df = pd.DataFrame(data=arr, columns=cols)

        sequences = self.store[f"{name}/SEQ"]

        for i, seq in enumerate(obj.sequences):
            seq.data.arr = sequences[i]
        return obj

    def parse_multiple(self, data):
        for d in data:
            self.parse_single(d)
        return data

    def close(self):
        self.store.close()
