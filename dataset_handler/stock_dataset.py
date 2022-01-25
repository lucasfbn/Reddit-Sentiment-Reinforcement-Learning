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

    def __init__(self):
        self.data = None

    def dump(self, root, dump_data=True):
        root = Path(root)

        meta_saver = MetaSaver(root / self.META_FN)
        meta_saver.dump(self.data)

        if dump_data:
            data_saver = DataSaver(root / self.DATA_FN)
            data_saver.dump_multiple(self.data)
            data_saver.close()

    def load_meta(self, root):
        root = Path(root)

        meta_parser = MetaParser(root / self.META_FN)
        self.data = meta_parser.parse()

    def load_data(self, root):
        root = Path(root)

        data_parser = DataParser(root / self.DATA_FN)
        self.data = data_parser.parse_multiple(self.data)
        data_parser.close()

    def load(self, root):
        root = Path(root)

        self.load_meta(root)
        self.load_data(root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        yield from self.data


class StockDatasetWandb(StockDataset):
    ARTIFACT_FN = "dataset"
    ARTIFACT_TYPE = "dataset"
    PATH_APPENDIX = "artefacts/dataset"

    def _get_root(self, run, fn, version, type):
        art = run.use_artifact(f'{run.entity}/{run.project}/{fn}:v{version}', type=type)
        return Path(art.download(root=os.getenv("WANDB_DIR") + self.PATH_APPENDIX))

    def wandb_load_meta_file(self, run_id, run):
        path = Path(wandb.restore(name=self.META_FN, run_path=f'{run.entity}/{run.project}/{run_id}',
                                  root=os.getenv("WANDB_DIR") + "files").name)
        path = path.parent  # because wandb adds the filename to the path
        self.load_meta(path)

    def wandb_load_data(self, run, version):
        root = self._get_root(run, fn=self.ARTIFACT_FN, version=version, type=self.ARTIFACT_TYPE)
        self.load_data(root)

    def wandb_load(self, run, version):
        root = self._get_root(run, fn=self.ARTIFACT_FN, version=version, type=self.ARTIFACT_TYPE)
        self.load_meta(root)
        self.load_data(root)

    def log_as_artifact(self, log_data=False):
        with tempfile.TemporaryDirectory() as tmpdirname:
            root = Path(tmpdirname)
            self.dump(root, dump_data=log_data)

            art = wandb.Artifact(self.ARTIFACT_FN, type=self.ARTIFACT_TYPE)
            art.add_file((root / self.META_FN).as_posix())

            if log_data:
                art.add_file((root / self.DATA_FN).as_posix())

            wandb.log_artifact(art)

    def log_as_file(self, run, log_data=False):
        root = Path(run.dir)
        self.dump(root, log_data)


if __name__ == "__main__":
    def non_wandb_usage():
        root = r"F:\wandb\artefacts\dataset"
        dataset = StockDataset()
        dataset.load_meta(root)
        dataset.load_data(root)
        # or
        # dataset.load(root)
        dataset.dump(root, dump_data=True)


    def wandb_usage():
        with wandb.init(project="TestsProject", group="Datasets") as run:
            dataset = StockDatasetWandb()
            dataset.wandb_load_meta_file("3e5jwups", run=run)
            dataset.wandb_load_data(run, 0)

            # or
            # dataset.wandb_load(run, 0)

            # dataset.log_as_file(run)
            # dataset.log_as_file(run, log_data=True)
            # dataset.log_as_artifact(log_data=True)
            print(dataset[0])


    def convert():
        with wandb.init(project="Trendstuff", group="Eval Stocks") as run:
            dataset = StockDatasetWandb()
            dataset.load_meta(".")
            dataset.log_as_file(run)


    # wandb_usage()
    convert()
