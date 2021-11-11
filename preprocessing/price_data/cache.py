import sqlite3

import pandas as pd

from utils import paths


class Cache:
    table = "data"
    ticker_name_col = "ticker"
    date_name_col = "date_day_shifted"

    def __init__(self, db_path=paths.price_data_cache):
        self.con = sqlite3.connect(db_path)
        self.cur = self.con.cursor()

    def append(self, df: pd.DataFrame, drop_duplicates=True):
        df = df.copy()  # make sure we don't alter the argument
        df = self._period_to_str(df)
        df.to_sql(self.table, con=self.con, if_exists="append")
        self.con.commit()

        if drop_duplicates:
            self.drop_duplicates()

    def _period_to_str(self, df):
        df[self.date_name_col] = df[self.date_name_col].astype(str)
        return df

    def _str_to_period(self, df):
        df[self.date_name_col] = pd.to_datetime(df[self.date_name_col]).dt.to_period("D")
        return df

    def drop_duplicates(self):
        cmd = (
            f"DELETE FROM {self.table} "
            "WHERE rowid NOT IN ( "
            "SELECT MIN(rowid) "
            f"FROM {self.table} "
            f"GROUP BY {self.ticker_name_col}, {self.date_name_col})"
        )
        self.cur.execute(cmd)
        self.con.commit()

    def get_all(self):
        cmd = f"SELECT * FROM {self.table}"
        df = pd.read_sql(sql=cmd, con=self.con).drop(columns=["index"])
        df = self._str_to_period(df)
        return df

    def get(self, ticker):
        cmd = f"SELECT * FROM {self.table} WHERE {self.ticker_name_col}='{ticker}'"
        df = pd.read_sql(sql=cmd, con=self.con).drop(columns=["index", "ticker"])
        df = self._str_to_period(df)
        return df

    def drop_tail(self, n):
        """
        Drops the n last entries of each ticker in the cache.
        Args:
            n: Number of entries to be deleted at the tail
        """
        df = self.get_all()
        df = df.sort_values(by=self.date_name_col)
        df = df.drop(df.groupby(["ticker"]).tail(n).index, axis=0)
        self._delete_all()
        self.append(df)

    def _delete_all(self):
        cmd = (
            f"DELETE FROM {self.table}"
        )
        self.cur.execute(cmd)
        self.con.commit()
