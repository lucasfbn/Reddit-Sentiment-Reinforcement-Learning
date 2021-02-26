import pandas as pd


class API:
    submission_schema = {"created_utc": "int64", "author": "object", "id": "object", "title": "object",
                         "subreddit": "object", "num_comments": "int64", "selftext": "object",
                         "score": "int64", "upvote_ratio": "float64", "full_link": "object",
                         "author_fullname": "object",
                         "allow_live_comments": "bool", "author_flair_css_class": "object",
                         "author_flair_richtext": "object", "author_flair_text": "object",
                         "author_flair_type": "object",
                         "author_patreon_flair": "object", "author_premium": "object", "awarders": "object",
                         "can_mod_post": "bool", "contest_mode": "bool", "domain": "object",
                         "gildings": "object", "is_crosspostable": "bool", "is_meta": "bool",
                         "is_original_content": "bool", "is_reddit_media_domain": "bool", "is_robot_indexable": "bool",
                         "is_self": "bool", "is_video": "bool", "link_flair_background_color": "object",
                         "link_flair_css_class": "object", "link_flair_richtext": "object",
                         "link_flair_template_id": "object", "link_flair_text": "object",
                         "link_flair_text_color": "object", "link_flair_type": "object", "locked": "bool",
                         "media_only": "bool", "no_follow": "bool", "num_crossposts": "int64", "over_18": "bool",
                         "parent_whitelist_status": "object", "permalink": "object", "pinned": "bool",
                         "post_hint": "object", "removed_by_category": "object",
                         "send_replies": "bool", "spoiler": "bool", "stickied": "bool",
                         "subreddit_id": "object", "subreddit_subscribers": "int64", "subreddit_type": "object",
                         "suggested_sort": "object", "thumbnail": "object", "total_awards_received": "int64",
                         "treatment_tags": "object",
                         "url": "object", "url_overridden_by_dest": "object",
                         "whitelist_status": "object", "created": "float64",
                         "author_flair_background_color": "object", "author_flair_text_color": "object",
                         "media": "object",
                         "media_embed": "object", "secure_media": "object", "secure_media_embed": "object",
                         "is_gallery": "object", "gallery_data": "object", "author_flair_template_id": "object",
                         "author_cakeday": "object", "banned_by": "object", "distinguished": "object",
                         "edited": "float64",
                         "gilded": "object", "collections": "object"}

    def to_df(self):
        if len(self.submissions) == 0:
            cols = [key for key in self.submission_schema]
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(self.submissions)
        df = df.astype(self.submission_schema)
        return df

    def _filter_removed(self, df):
        cols_to_check_if_removed = ["author", "selftext", "title"]
        for col in cols_to_check_if_removed:
            df = df[~df[col].isin(["[removed]", "[deleted]"])]
        return df
