import logging

import praw
from psaw import PushshiftAPI
import pandas as pd
import preprocess.sentiment_analysis.download.from_pushshift.schemas as schema

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
for logger_name in ("praw", "prawcore"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


class Graber:

    def __init__(self, use_praw=False):
        self.use_praw = use_praw

        if use_praw:
            reddit_api = praw.Reddit(user_agent="RedditTrend",
                                     client_id="XGPKdr-omVKy6A",
                                     client_secret="tXPhgIcvevtWoJzwFQXYXgyL5bF9JQ",
                                     username="qrzte",
                                     password="ga5FwRaXcRyJ77e")
            self.pushshift_api = PushshiftAPI(reddit_api)
        else:
            self.pushshift_api = PushshiftAPI()

    def _extract_data(self, r):
        if self.use_praw:
            data = self._extract_data_praw(r)
        else:
            data = self._extract_data_pushshift(r)
        return data

    def _extract_data_pushshift(self, r):
        temp = r.d_
        new_temp = {}
        for key, value in schema.submissions.items():
            try:
                new_temp[key] = temp[key]
            except KeyError:
                new_temp[key] = None

        return new_temp

    def _extract_data_praw(self, r):

        ignore_keys = ["_replies", "_submission", "_reddit", "_fetched", "all_awardings", "author", "subreddit",
                       "gildings", "media_metadata", "user_reports", "treatment_tags", "mod_reports",
                       "subreddit_type", "collapsed_because_crowd_control", "subreddit_name_prefixed", "awarders",
                       "author_flair_richtext", "link_flair_richtext", "media_embed", "secure_media_embed",
                       "_comments_by_id"]
        temp = r.__dict__
        temp["author_id"] = None
        try:
            temp["author_id"] = r.author.id
        except:
            pass
        temp["sub_id"] = r.subreddit.id

        for ik in ignore_keys:
            temp.pop(ik, None)
        return temp

    def _handler(self, start, end, kind, **kwargs):
        diff = divmod((end - start).total_seconds(), 3600)[0]
        if diff >= 24:
            intermediate_data = []
        else:
            return self._get(start, end, kind, **kwargs)

    def _get(self, start, end, kind, **kwargs):
        start, end = int(start.timestamp()), int(end.timestamp())

        if kind == "comments":
            response = self.pushshift_api.search_comments(after=start, before=end, **kwargs)

            data = []
            for r in response:
                data.append(self._extract_data(r))
        elif kind == "submissions":
            response = self.pushshift_api.search_submissions(after=start, before=end, **kwargs)

            data = []
            for r in response:
                data.append(self._extract_data(r))
        else:
            raise ValueError("Invalid kind. Either comments or submissions.")
        return data

    def _postprocess(self, data, kind):
        if kind == "submissions":
            df = pd.DataFrame(data)
            df = df.astype(schema.submissions)

        return df

    def get_comments(self, start, end, **kwargs):
        return self._get(start, end, "comments", **kwargs)

    def get_submissions(self, start, end, **kwargs):
        data = self._get(start, end, "submissions", **kwargs)
        return self._postprocess(data, "submissions")
