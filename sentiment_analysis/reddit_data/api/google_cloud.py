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
        log.info("Starting upload...")
        try:
            df.to_gbq(destination_table=f"{dataset}.{table}",
                      project_id=self.project_id,
                      if_exists="append")
        except GenericGBQException as e:
            log.warn(f"Error while uploading. Error:\n {e}. \n Len df: {len(df)}")

    def download(self, start, end, fields, sql=None, check_duplicates=True):
        log.info("Starting download...")
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

    def detect_gaps(self, start=None, end=None, save_json=True):
        log.debug("Detecting gaps...")
        start, end = dt_to_timestamp(start), dt_to_timestamp(end)

        sql = f"""SELECT created_utc FROM `redditdata-305217.data.submissions`"""

        if start is not None and end is not None:
            sql += f" WHERE created_utc BETWEEN {start} AND {end}"
        elif start is not None and end is None:
            sql += f" WHERE created_utc >= {start}"
        elif end is not None and start is None:
            sql += f" WHERE created_utc <= {end}"

        df = self.download(None, None, None, sql, check_duplicates=False)

        # df.to_csv("gaps.csv", sep=";", index=False)
        # df = pd.read_csv("gaps.csv", sep=";")

        date_full = pd.to_datetime(df["created_utc"], unit="s")
        date_full = date_full.dt.tz_localize("UTC")
        date_mesz = date_full.dt.tz_convert("Europe/Berlin")
        df["start"] = date_mesz.dt.to_period('H').dt.to_timestamp()
        date_list = df["start"].tolist()

        min_date = df["start"].min()
        max_date = df["start"].max()

        daterange = pd.date_range(start=min_date, end=max_date - pd.Timedelta(hours=1), freq="H")
        daterange = daterange.tolist()

        difference = list(set(daterange) - set(date_list))
        difference = pd.Series(difference)
        difference = difference.sort_values()
        difference = difference.tolist()

        last_diff = None
        periods = []
        sub_period = []
        for diff in difference:
            if last_diff is None:
                last_diff = diff
                sub_period.append(diff)
            else:
                current_diff = diff - last_diff
                if current_diff.seconds > 3600:
                    periods.append(sub_period)
                    sub_period = [diff]
                else:
                    sub_period.append(diff)
            last_diff = diff

        if sub_period:
            periods.append(sub_period)

        min_max_subperiod = [{"start": start, "end": end, "min_date": str(min_date), "max_date": str(max_date)}]
        for sub_period in periods:
            min_max_subperiod.append({"min": str(sub_period[0]), "max": str(sub_period[len(sub_period) - 1])})

        if save_json:
            with open("gaps.json", "w+") as f:
                json.dump(min_max_subperiod, f)
        return min_max_subperiod


if __name__ == '__main__':
    start = datetime(year=2021, month=3, day=18)
    end = datetime(year=2021, month=3, day=25)
    # db = BigQueryDB()
    # df = db.download(start, end,
    #                  fields=["author", "created_utc", "id", "num_comments", "title", "selftext", "subreddit"],
    #                  check_duplicates=False)
    # print()
    # df.to_csv("raw.csv", sep=";", index=False)

    db = BigQueryDB()
    db.detect_gaps(start=start, end=end)
    # db.upload(pd.DataFrame({"one": [1, 2, 3]}), dataset="data", table="test_subm1")
