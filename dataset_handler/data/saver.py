import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


class Saver:

    def __init__(self, fp):
        self.fp = fp
        self.store = h5py.File(self.fp, "w")

    def dump_single(self, obj):
        name = obj.name

        self.store.create_dataset(f"{name}/COLS", data=np.array(list(obj.df.columns), dtype="S"))
        self.store.create_dataset(f"{name}/DF", data=obj.df.to_numpy(dtype="float32"))

        sequences = tuple(seq.data.arr for seq in obj.sequences.lst)
        seq_stacked = np.stack(sequences, axis=0)

        self.store.create_dataset(f"{name}/SEQ", data=seq_stacked)

    def dump_multiple(self, data: list):
        for d in tqdm(data, desc="DataSaver"):
            self.dump_single(d)

    def close(self):
        self.store.close()
