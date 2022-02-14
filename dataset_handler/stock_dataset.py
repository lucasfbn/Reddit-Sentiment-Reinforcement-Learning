import copy
import os
import tempfile
from pathlib import Path

import pandas as pd
import wandb

from dataset_handler.classes.ticker import Sequences
from dataset_handler.data.parser import Parser as DataParser
from dataset_handler.data.saver import Saver as DataSaver
from dataset_handler.meta.parser import Parser as MetaParser
from dataset_handler.meta.saver import Saver as MetaSaver


class StockDataset:
    META_FN = "meta.json"
    DATA_FN = "data.h5"

    def __init__(self, parse_date=False):
        self.parse_date = parse_date
        self.data = None

        self.stats = None
        self.index = None

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

        meta_parser = MetaParser(root / self.META_FN, parse_date=False)
        self.data = meta_parser.parse()

        if self.parse_date:
            self._parse_dates()

        self.stats = Statistics(self.data, self.parse_date)
        self.index = Indexer(self.data, self.parse_date)

    def load_data(self, root):
        root = Path(root)

        data_parser = DataParser(root / self.DATA_FN)
        self.data = data_parser.parse_multiple(self.data)
        data_parser.close()

    def _update_attributes(self):
        self.index = Indexer(self.data, self.parse_date)
        self.stats = Statistics(self.data, self.parse_date)

    def filter_min_len(self, min_len):
        self.data = [obj for obj in self.data if len(obj) > min_len]
        self._update_attributes()

    def _parse_dates(self):
        _ = [obj.sequences.parse_dates() for obj in self.data]

    def is_empty(self):
        return all(obj.is_empty() for obj in self.data)

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


class Statistics:

    def __init__(self, data, parse_date):
        self.data = data
        self.parse_date = parse_date

    def min_max_date(self):

        min_ = pd.Period("01/01/3000")
        max_ = pd.Period("01/01/1000")

        for obj in self.data:
            obj_min, obj_max = obj.sequences.min_max_dates()

            if obj_min < min_:
                min_ = obj_min

            if obj_max > max_:
                max_ = obj_max

        return min_, max_

    def date_distribution(self):
        dfs = [obj.sequences.to_df() for obj in self.data]
        df = pd.concat(dfs)

        if not self.parse_date:
            df["date"] = pd.to_datetime(df["date"]).dt.to_period("d")

        df = df.sort_values(by=["date"])
        vc = df["date"].value_counts(sort=False)
        return pd.DataFrame({"date": vc.index, "count": vc}).reset_index(drop=True)


class Indexer:

    def __init__(self, data, parse_date):
        self.data = data
        self.parse_date = parse_date

    def __getitem__(self, idx):
        min_date, max_date = idx.start, idx.stop

        new_data = []

        for obj in self.data:
            new_obj = copy.copy(obj)

            df = obj.sequences.to_df()

            if not self.parse_date:
                df["date"] = pd.to_datetime(df["date"]).dt.to_period("d")

            slice_ = slice(None, None)

            if min_date is None:
                df = df[df["date"] <= max_date]
                if len(df) > 0:
                    slice_ = slice(None, len(df), None)
            elif max_date is None:
                df = df[df["date"] >= min_date]
                if len(df) > 0:
                    slice_ = slice(int(df.index[0]), None, None)
            else:
                df = df[(min_date <= df["date"]) & (df["date"] <= max_date)]
                if len(df) > 0:
                    slice_ = slice(int(df.index[0]), int(df.index[len(df) - 1]) + 1, None)

            new_obj.sequences = Sequences()
            new_obj.sequences.lst = [] if len(df) == 0 else obj.sequences[slice_]
            new_data.append(new_obj)

        new_dataset = StockDataset(self.parse_date)
        new_dataset.data = new_data
        new_dataset.index = Indexer(new_data, self.parse_date)
        new_dataset.stats = Statistics(new_data, self.parse_date)
        new_dataset.filter_min_len(1)
        return new_dataset


class StockDatasetWandb(StockDataset):
    ARTIFACT_FN = "dataset"
    ARTIFACT_TYPE = "dataset"
    PATH_APPENDIX = "artefacts/dataset"

    def _get_root(self, run, fn, version, type):
        art = run.use_artifact(f'{run.entity}/{run.project}/{fn}:v{version}', type=type)
        return Path(art.download(root=os.getenv("WANDB_DIR") + self.PATH_APPENDIX))

    def wandb_load_meta_file(self, run_id, run):
        path = Path(wandb.restore(name=self.META_FN, run_path=f'{run.entity}/{run.project}/{run_id}',
                                  root=f"{os.getenv('WANDB_DIR')}run_files/{run_id}").name)
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
        dataset = StockDataset(parse_date=True)
        dataset.load_meta(root)
        dataset.load_data(root)

        # max_date_ds = dataset.index[:pd.Period("2021-04-23")]
        #
        # min_date_ds = dataset.index[pd.Period("2021-04-23"):]
        #
        # in_date_ds = dataset.index[pd.Period("2021-04-23"):pd.Period("2021-06-13")]

        # or
        # dataset.load(root)
        # dataset.dump(root, dump_data=True)


    def wandb_usage():
        with wandb.init(project="Trendstuff", group="Throwaway") as run:
            dataset = StockDatasetWandb()
            dataset.wandb_load_meta_file("2e779tzl", run=run)
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


    non_wandb_usage()
    # wandb_usage()
    # convert()
