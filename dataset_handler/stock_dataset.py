import os
import tempfile
from pathlib import Path

import wandb

from dataset_handler.data.parser import Parser as DataParser
from dataset_handler.data.saver import Saver as DataSaver
from dataset_handler.meta.parser import Parser as MetaParser
from dataset_handler.meta.saver import Saver as MetaSaver


class StockDataset:
    META_FN = "meta.json"
    DATA_FN = "data.h5"

    def __init__(self, root):
        self.root = Path(root)

        self._meta_parser = None
        self._meta_saver = None
        self._data_parser = None
        self._data_saver = None

        self._data = None

    @property
    def data(self):
        return self._data

    def dump(self, dump_data=True):
        meta_saver = MetaSaver(self.root / self.META_FN)
        meta_saver.dump(self._data)

        if dump_data:
            data_saver = DataSaver(self.root / self.DATA_FN)
            data_saver.dump_multiple(self._data)
            data_saver.close()

    def load_meta(self):
        if self._meta_parser is None:
            self._meta_parser = MetaParser(self.root / self.META_FN)

        self._data = self._meta_parser.parse()

    def _init_parser(self):
        if self._data_parser is None:
            self._data_parser = DataParser(self.root / self.DATA_FN)

    def _close_parser(self):
        if self._data_parser is not None:
            self._data_parser.close()

    def load_data(self):
        self._init_parser()
        self._data = self._data_parser.parse_multiple(self._data)
        self._close_parser()

    def load(self):
        self.load_meta()
        self.load_data()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self):
        yield from self._data


class StockDatasetWandb(StockDataset):
    ARTIFACT_FN = "dataset"
    ARTIFACT_TYPE = "dataset"
    PATH_APPENDIX = "artefacts/dataset"

    def __init__(self, run, version, root=None):

        if root is None:
            root = self._get_root(run, fn=self.ARTIFACT_FN, version=version, type=self.ARTIFACT_TYPE)

        super().__init__(root)

    def _get_root(self, run, fn, version, type):
        art = run.use_artifact(f'{run.entity}/{run.project}/{fn}:v{version}', type=type)
        return Path(art.download(root=os.getenv("WANDB_DIR") + self.PATH_APPENDIX))

    def wandb_load(self, load_data=False):
        self.load_meta()
        if load_data:
            self.load_data()

    def log_as_artifact(self, log_data=False):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.root = Path(tmpdirname)
            self.dump(dump_data=log_data)

            art = wandb.Artifact(self.ARTIFACT_FN, type=self.ARTIFACT_TYPE)
            art.add_file((self.root / self.META_FN).as_posix())

            if log_data:
                art.add_file((self.root / self.DATA_FN).as_posix())

            wandb.log_artifact(art)

    def log_as_file(self, run, log_data=False):
        self.root = Path(run.dir)
        self.dump(log_data)


if __name__ == "__main__":
    def non_wandb_usage():
        dataset = StockDataset(r"F:\wandb\artefacts\dataset")
        dataset.load_meta()
        dataset.load_data()
        dataset.dump(dump_data=True)


    def wandb_usage():
        with wandb.init(project="Trendstuff", group="Datasets") as run:
            dataset = StockDatasetWandb(run, 0)
            dataset.wandb_load()
            # # dataset.log_as_file(run, log_data=True)
            # dataset.log_as_artifact(log_data=True)
            print(dataset[0])


    wandb_usage()
