import pandas as pd

from datetime import datetime
from utils import dt_to_timestamp


class BigQueryDB:

    def __init__(self):
        self.project_id = "redditdata-305217"

    def auth(self):
        self.upload(pd.DataFrame({"test": [1, 2, 3]}), dataset="auth", table="auth_table")

    def upload(self, df, dataset, table):
        df.to_gbq(destination_table=f"{dataset}.{table}",
                  project_id=self.project_id,
                  if_exists="append")

    def download(self, start, end, fields):
        start, end = dt_to_timestamp(start), dt_to_timestamp(end)

        sql = f"""
        SELECT {", ".join(fields)} FROM `redditdata-305217.data.submissions`
        WHERE created_utc BETWEEN {start} AND {end}
        """

        return pd.read_gbq(sql, project_id=self.project_id)


if __name__ == '__main__':
    # start = datetime(year=2021, month=1, day=13)
    # end = datetime(year=2021, month=1, day=13, hour=1)
    # db = BigQueryDB()
    # df = db.download(start, end, fields=["author", "created_utc", "id", "num_comments", "title", "selftext", "subreddit"])
    # df.to_csv("raw.csv", sep=";", index=False)

    db = BigQueryDB()
    db.auth()
    # db.upload(pd.DataFrame({"one": [1, 2, 3]}), dataset="data", table="test_subm1")
