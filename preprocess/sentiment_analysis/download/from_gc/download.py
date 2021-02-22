from preprocess.sentiment_analysis.db.db_handler import DB
from datetime import datetime


def download(start, end, fields=["author", "created_utc", "id", "num_comments", "title", "selftext", "subreddit"]):
    start = int(start.timestamp())
    end = int(end.timestamp())

    sql = f"""
    SELECT {", ".join(fields)} FROM `redditdata-305217.data.submissions`
    WHERE created_utc BETWEEN {start} AND {end}
    """

    database = DB()
    df = database.down(sql)
    return df


if __name__ == '__main__':
    start = datetime(year=2021, month=1, day=13)
    end = datetime(year=2021, month=1, day=13, hour=1)
    df = download(start, end)
    df.to_csv("raw.csv", sep=";", index=False)
