import pickle as pkl

import pandas as pd

from utils import drop_stats, log


class Preprocessor:

    def __init__(self,
                 author_blacklist,
                 cols_to_check_if_removed,
                 cols_to_be_cleaned,
                 max_subm_p_author_p_day,
                 filter_authors,
                 path=None,
                 df=None):

        if path is None and df is None:
            raise ValueError("Specify either a path or a df.")

        if path is not None:
            self.df = pd.read_csv(path, sep=";")
        else:
            self.df = df

        self.author_blacklist = author_blacklist
        self.cols_to_check_if_removed = cols_to_check_if_removed
        self.cols_to_be_cleaned = cols_to_be_cleaned
        self.max_subm_p_author_p_day = max_subm_p_author_p_day
        self.grps = []
        self.filter_authors = filter_authors

    def _add_date_col(self):
        self.df["date_full"] = pd.to_datetime(self.df["created_utc"], unit="s")

        self.df["date_full"] = self.df["date_full"].dt.tz_localize("UTC")
        self.df["date_mesz"] = self.df["date_full"].dt.tz_convert("Europe/Berlin")
        self.df["start"] = self.df["date_mesz"].dt.to_period('H')
        self.df["start_timestamp"] = self.df["date_mesz"].astype(int) / 10 ** 9

        end = self.df["date_mesz"] + pd.Timedelta(seconds=3600)
        self.df["end"] = end.dt.to_period('H')
        self.df["end_timestamp"] = end.astype(int) / 10 ** 9

        self.df["date"] = self.df["date_mesz"].dt.to_period('D')

    @drop_stats
    def _filter_removed(self):
        for col in self.cols_to_check_if_removed:
            self.df = self.df[~self.df[col].isin(["[removed]", "[deleted]"])]

    @drop_stats
    def _filter_authors(self):

        if not self.filter_authors:
            return

        filtered_rows = []

        cols = self.df.columns
        grps = self.df.groupby(["date"])

        for i, (name, grp) in enumerate(grps):
            grp = pd.DataFrame(grp)
            authors = grp.groupby(["author"])

            for j, (author_id, author_grp) in enumerate(authors):
                log.debug(f"Processing grp: {i}/{len(grps)}. Author: {j}/{len(authors)}")

                if author_id in self.author_blacklist:
                    continue

                if len(author_grp) == self.max_subm_p_author_p_day:
                    filtered_rows.append(author_grp)
                else:
                    author_grp = author_grp.sort_values(by=["num_comments"], ascending=False)
                    filtered_rows.append(author_grp.head(self.max_subm_p_author_p_day))

        log.debug("Starting to concatenate filtered_rows.")
        self.df = pd.concat(filtered_rows)
        self.df.columns = cols

    def _delete_non_alphanumeric(self):
        for col in self.cols_to_be_cleaned:
            self.df[col] = self.df[col].str.replace('[^\w\s,.?!()-+:"]', '', regex=True)

    def _groupby_date(self):
        grps = self.df.groupby("start")

        new_grps = []
        for name, grp in grps:
            new_grps.append({"id": str(name), "df": grp,
                             "start": grp["start"].iloc[0], "end": grp["end"].iloc[0],
                             "start_timestamp": grp["start_timestamp"].iloc[0],
                             "end_timestamp": grp["end_timestamp"].iloc[0],
                             "subreddit": "all"})

        self.grps = new_grps

    def exec(self):
        self._filter_removed()
        self._add_date_col()
        self._filter_authors()
        self._delete_non_alphanumeric()
        self._groupby_date()
        return self.grps


if __name__ == '__main__':
    prep = Preprocessor(author_blacklist=[],
                        cols_to_check_if_removed=["author", "selftext", "title"],
                        cols_to_be_cleaned=["title"],
                        max_subm_p_author_p_day=1,
                        filter_authors=True,
                        path="gc_dump.csv")
    grps = prep.exec()
    with open("raw.pkl", "wb") as f:
        pkl.dump(grps, f)
