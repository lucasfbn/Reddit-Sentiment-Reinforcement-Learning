from dataset_handler.stock_dataset import StockDataset


class Split:

    def __init__(self, dataset: StockDataset):
        self.dataset = dataset
        self._date_distro = self._get_distr()

    def _get_distr(self):
        return self.dataset.stats.date_distribution()

    def _find_abs_vals(self, splits):

        assert sum(splits) == 1.0

        total = self._date_distro["count"].sum()

        splits_vals = []
        last = None
        for splt in splits:

            splt_val = int(splt * total)
            if last is not None:
                splt_val += last
            last = splt_val

            splits_vals.append(splt_val)
        return splits_vals

    def _find_split_idx(self, splits_vals):

        idxs = []

        self._date_distro["sum"] = self._date_distro["count"].expanding(min_periods=0).sum()

        for i, spl_val in enumerate(splits_vals):
            temp = self._date_distro[self._date_distro["sum"] <= spl_val]

            start = 0 if i == 0 else idxs[-1][1] + 1
            end = temp.tail(1).index[0] if i != len(splits_vals) - 1 else self._date_distro.tail(1).index[0]
            idxs.append((start, end))

        return idxs

    def _match_idx_date(self, idxs):
        return [(self._date_distro.at[start, "date"], self._date_distro.at[end, "date"]) for start, end in idxs]

    def _slice(self, idxs):
        return [self.dataset.index[start:end] for start, end in idxs]

    def split(self, splits):
        splits_vals = self._find_abs_vals(splits)
        splits_idxs = self._find_split_idx(splits_vals)
        splits_dates = self._match_idx_date(splits_idxs)
        return self._slice(splits_dates)


def split(dataset: StockDataset, splits: list):
    splitter = Split(dataset)
    return splitter.split(splits)


if __name__ == "__main__":
    root = r"F:\wandb\artefacts\dataset"
    dataset = StockDataset(parse_date=False)
    dataset.load_meta(root)
    splits = split(dataset, [0.25, 0.25, 0.25, 0.25])
    distros = [d.stats.date_distribution()["count"].sum() for d in splits]
    print(distros)
