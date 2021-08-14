import json
from datetime import datetime

import pandas as pd
from google.cloud import bigquery

from pandas_gbq.gbq import GenericGBQException

from utils.util_funcs import dt_to_timestamp, log


class BigQueryDB:

    def __init__(self):
        self.project_id = "redditdata-305217"

    def auth(self):
        self.upload(pd.DataFrame({"test": [1, 2, 3]}), dataset="auth", table="auth_table")

    def upload(self, df, dataset, table):
        log.debug("Starting upload...")
        try:
            df.to_gbq(destination_table=f"{dataset}.{table}",
                      project_id=self.project_id,
                      if_exists="append")
        except GenericGBQException as e:
            log.warn(f"Error while uploading. Error:\n {e}. \n Len df: {len(df)}")

    def download(self, start, end, fields, sql=None, check_duplicates=True):
        log.debug("Starting download...")
        start, end = dt_to_timestamp(start), dt_to_timestamp(end)

        if sql is None:
            sql = f"""
            SELECT {", ".join(fields)} FROM `redditdata-305217.data.submissions`
            WHERE created_utc BETWEEN {start} AND {end}
            """

        client = bigquery.Client()
        df = client.query(sql, project=self.project_id).to_dataframe()

        if check_duplicates:
            df = df.drop_duplicates(subset=['id'])
        return df


class DetectGaps:

    def __init__(self, df):
        self.df = df

        self._min_date = None
        self._max_date = None

        self._daterange = None
        self._diff = None

    def _assign_date(self):
        self.df["temp_date"] = pd.to_datetime(self.df["created_utc"], unit="s")
        self.df["temp_date"] = self.df["temp_date"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
        self.df["temp_date_period"] = self.df["temp_date"].dt.to_period('H').dt.to_timestamp()
        self.df = self.df.sort_values(by=["temp_date_period"])

        self._min_date = self.df["temp_date_period"].min()
        self._max_date = self.df["temp_date_period"].max()

    def _generate_date_range(self):
        self._daterange = pd.date_range(start=self._min_date, end=self._max_date - pd.Timedelta(hours=1), freq="H")

    def _find_diff(self):
        diff = list(set(self._daterange.tolist()) - set(self.df["temp_date_period"].tolist()))
        diff = pd.Series(diff)
        diff = diff.sort_values()
        diff = diff.tolist()
        self._diff = diff

    def _make_report(self):
        self._diff = [str(x) for x in self._diff]

    def run(self):
        self._assign_date()
        self._generate_date_range()
        self._find_diff()
        self._make_report()
        return self._diff
