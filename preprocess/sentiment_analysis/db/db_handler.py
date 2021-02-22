import pandas as pd


class DB:

    def __init__(self):
        self.project_id = "redditdata-305217"

    def up(self, df, dataset, table):
        # df = pd.DataFrame(data)
        df.to_gbq(destination_table=f"{dataset}.{table}",
                  project_id=self.project_id,
                  if_exists="append")

    def down(self, sql):
        return pd.read_gbq(sql, project_id=self.project_id)
