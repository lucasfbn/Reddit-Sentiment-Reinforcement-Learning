import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from preprocessing.preprocessing_utils.preprocessor import Preprocessor
from preprocessing.preprocessing_utils.stock_prices import StockPrices

pd.options.mode.chained_assignment = None


class MergePreprocessing(Preprocessor):

    def __init__(self,
                 start_hour, start_min,
                 market_symbols=[],
                 start_offset=30,
                 fill_gaps=True,
                 scale_cols_daywise=True,
                 live=False,
                 limit=None):

        self.df = self.load(self.fn_initial, initial=True)
        self.grps = []
        self.start_hour = start_hour
        self.start_min = start_min
        self.market_symbols = market_symbols
        self.start_offset = start_offset
        self.fill_gaps = fill_gaps
        self.scale_cols_daywise = scale_cols_daywise
        self.live = live
        self.limit = limit

    def _handle_time(self):
        self.df["time"] = pd.to_datetime(self.df["end"], format="%Y-%m-%d %H:%M")
        self.df = self.df.sort_values(by=["time"])
        self.df["time_shifted"] = self.df["time"] - pd.Timedelta(hours=self.start_hour,
                                                                 minutes=self.start_min) + pd.Timedelta(days=1)

        self.df["date_day"] = pd.to_datetime(self.df['time_shifted']).dt.to_period('D')

    def _filter_market_symbol(self):
        if self.market_symbols:
            self.df = self.df[self.df["market_symbol"].isin(self.market_symbols)]

    def _scale_daywise(self):
        if not self.scale_cols_daywise:
            return

        scaled = pd.DataFrame()
        grp_by = self.df.groupby(["date_day"])
        for name, group in grp_by:
            scaler = MinMaxScaler()

            group[[col + "_scaled_daywise" for col in self.cols_to_be_scaled_daywise]] \
                = scaler.fit_transform(group[self.cols_to_be_scaled_daywise])
            scaled = scaled.append(group)
        self.df = scaled

    def _grp_by(self):
        grp_by = self.df.groupby(["ticker"])

        for name, group in grp_by:
            self.grps.append({"ticker": name, "data": group.groupby(["date_day"]).agg("sum").reset_index()})

    def _limit(self):
        if self.limit is not None:
            self.grps = self.grps[:self.limit]

    def _drop_short(self):
        filtered_grps = []

        for grp in self.grps:
            if len(grp["data"]) > self.min_len:
                filtered_grps.append(grp)
        self.grps = filtered_grps

    def _mark_available(self):
        """
        Used during training. Marks the entries that should be taken into account while training. For instance, if we
        set min_len to 2, we, in reality, are not able to trade the instances prior to the entry which fulfills the
        min len.
        Example:
            min_len = 3
            04.01 - first entry, NOT tradeable
            05.01 - second entry, NOT tradeable
            06.01 - third entry, IS tradeable

        E.g. the non tradeable entries will be taken into account historically but will not be used as individual
         sequences.

        :return:
        """
        for grp in self.grps:
            # Basically a list where index 0 to min_len is False and the rest is True
            available = [False] * self.min_len + [True] * (len(grp["data"]) - self.min_len)
            grp["data"]["available"] = available

    def _handle_gaps(self):
        """
        Handles gaps in between the first and the last occurrence. E.g. if a whole day (or more) is missing during a
        certain period these days will be added and their values will be set to 0.
        """

        if not self.fill_gaps:
            return

        raise NotImplemented("HANDLE GAPS CURRENTLY FILLS NANS WITH 0. THIS IS NOT COMPATIBLE WITH THE CURRENT"
                             "IMPLEMENTATION OF THE 'available' COLUMN.")

        for grp in self.grps:
            df = grp["data"]
            min_date = df["date_day"].min().to_timestamp()
            max_date = df["date_day"].max().to_timestamp()

            # Create temp df with the daterange between min and max date in original df
            temp = pd.DataFrame()
            temp["date_day"] = pd.date_range(start=min_date, end=max_date)
            temp["date_day"] = temp["date_day"].dt.to_period("D")

            df = df.merge(temp, on="date_day", how="outer")
            df = df.sort_values(by=["date_day"])
            df = df.fillna(0)
            grp["data"] = df

    def _add_stock_prices(self):
        new_grps = []

        for i, grp in enumerate(self.grps):
            print(f"Processing {i}/{len(self.grps)}")
            sp = StockPrices(grp, start_offset=self.start_offset, live=self.live)
            new_grps.append({"ticker": grp["ticker"], "data": sp.download()})

        self.grps = new_grps

    def _sort_chronologically(self):
        for grp in self.grps:
            grp["data"]["date"] = grp["data"]['date_day'].dt.to_timestamp('s')
            grp["data"] = grp["data"].sort_values(by=["date"])

    def _backfill_availability(self):
        for grp in self.grps:

            # If the first element in the available column is nan we set it to False. Since we use ffill to fill the
            # nans we need a value in the first row.
            first_availability = grp["data"]["available"].iloc[0]
            if first_availability != True:
                grp["data"]["available"].iloc[0] = False

            grp["data"]["available"] = grp["data"]["available"].fillna(method="ffill")

    def pipeline(self):
        self._handle_time()
        self._filter_market_symbol()
        self._scale_daywise()
        self._grp_by()
        self._drop_short()
        self._mark_available()
        self._limit()
        self._handle_gaps()
        self._add_stock_prices()
        self._sort_chronologically()
        self._backfill_availability()
        self.save(self.grps, self.fn_merge_hype_price)
