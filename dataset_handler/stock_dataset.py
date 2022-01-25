from dataset_handler.meta.parser import Parser as MetaParser
from dataset_handler.data.parser import Parser as DataParser
from dataset_handler.meta.saver import Saver as MetaSaver
from dataset_handler.data.saver import Saver as DataSaver
from pathlib import Path
import os


class StockDataset:

    def __init__(self, root):
        self.root = Path(root)

        self._meta_parser = None
        self._meta_saver = None
        self._data_parser = None
        self._data_saver = None

        self._data = None

    def dump(self, data, dump_data=True):
        meta_saver = MetaSaver(self.root / "meta.h5")
        meta_saver.dump(data)

        if dump_data:
            data_saver = DataSaver(self.root / "data.h5")
            data_saver.dump_multiple(data)
            data_saver.close()

    def load_meta(self):
        if self._meta_parser is None:
            self._meta_parser = MetaParser(self.root / "meta.h5")

        self._data = self._meta_parser.parse()

    def _init_parser(self):
        if self._data_parser is None:
            self._data_parser = DataParser(self.root / "data.h5")

    def load_data_single(self, idx):
        self._init_parser()
        self._data[idx] = self._data_parser.parse_single(self._data[idx])
        return self._data[idx]

    def load_data(self):
        self._init_parser()
        self._data = self._data_parser.parse_multiple(self._data)
        self.close_parser()
        return self._data

    def close_parser(self):
        if self._data_parser is not None:
            self._data_parser.close()


class StockDatasetWandb(StockDataset):

    def download(self, run, fn, version, type):
        art = run.use_artifact(f'{run.entity}/{run.project}/{fn}:v{version}', type=type)
        self.root = Path(art.download(root=os.getenv("WANDB_DIR") + "artefacts/dataset"))

    def log(self):
        art = wandb.Artifact("dataset", type="dataset")
        art.add_dir(self.root)
        wandb.log_artifact(art)


import pickle as pkl
import wandb

# path = r"F:\wandb\artefacts\dataset.pkl"
# # path = r"temp3.pkl"
#
# with open(path, "rb") as f:
#     data = pkl.load(f)
import time

dataset = StockDataset(r"F:\wandb\artefacts\dataset")

start = time.time()
# dataset.dump(data, dump_data=True)
dataset.load_meta()
print(time.time() - start)
dataset.load_data()
print(time.time() - start)

time.sleep(100)

print()
