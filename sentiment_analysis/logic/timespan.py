class Timespan:

    def __init__(self, start, end, df, subreddit="all"):
        self.subreddit = subreddit
        self.df = df
        self.end = end
        self.start = start

    def __eq__(self, other):
        if not isinstance(other, Timespan):
            raise NotImplementedError

        if (self.subreddit == other.subreddit and
                self.df.reset_index(drop=True).equals(other.df.reset_index(drop=True)) and
                self.end == other.end and
                self.start == other.start):
            return True
        return False
