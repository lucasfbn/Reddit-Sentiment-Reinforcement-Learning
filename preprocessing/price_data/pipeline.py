import pandas as pd
from pandas import Period

from preprocessing.price_data.cache import Cache


def add_placeholder_entry():
    """
    Adds a placeholder entry to the sql db to define the schema. This is useful when initializing the db.
    """
    c = Cache()
    df = pd.DataFrame({'ticker': ["Placeholder"],
                       "Close": [1.0], "Open": [1.0], "Volume": [1], "Adj_Close": [1.0],
                       "High": [1.0], "Low": [1.0], "date_day": [Period('2021-05-10', 'D')]})
    c.append(df)


add_placeholder_entry()
