import pandas as pd

from preprocessing.sequence import Sequence, Metadata, Data
from preprocessing.ticker import Sequences


class SequenceGenerator:
    date_col_name = "date_day_shifted"

    def __init__(self, df: pd.DataFrame, sequence_len: int, include_available_days_only: bool,
                 price_column: str = None, exclude_cols_from_sequence: list = [], ticker_name: str = ""):
        """
        Generates sequences from a given dataframe.

        Args:
            df: Input dataframe
            sequence_len: Length of the desired sequence
            include_available_days_only: Whether to filter sequences which were not available for trading
            price_column: Name of the last column (in most cases: "price"). The NN will use the last column of a sequence
             as the price, therefore we have to ensure that the last column is the correct one
            exclude_cols_from_sequence: Cols that may be used during the generation of sequences but are not subject of
             the final sequences and can therefore be dropped
        """
        self.df = df
        self.ticker_name = ticker_name
        self.sequence_len = sequence_len
        self.include_available_days_only = include_available_days_only
        self.price_column = price_column
        self.exclude_cols_from_sequence = exclude_cols_from_sequence

        self._sliced = []
        self._sequences = []

    def slice_sequences(self) -> list:
        """
        Slices several parts out of a df - based on the sequence len.

        Example (in non-dataframe format) with sequence_len = 3:
        1, 2, 3, 4, 5, 6, 7
        -> [[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7]]
        """
        for i in range(len(self.df) - self.sequence_len + 1):
            sliced = self.df[i: i + self.sequence_len]
            self._sliced.append(sliced)
        return self._sliced

    @staticmethod
    def _add_availability(df):
        """
        We take the last element as a reference since this is the "current" element in this case - the elements prior
        to the last element are "past" elements. For instance, if we look at wednesday and have the data from monday
        and tuesday - it doesn't matter whether it was possible to trade on monday or tuesday, all that matters is that
        we can trade on wednesday.
        """
        return bool(df["available"].iloc[len(df) - 1])

    def _add_price(self, df):
        """
        See @_add_availability
        """
        return float(df[self.price_column].iloc[len(df) - 1])

    def _add_raw_price(self, df):
        return float(df["price_raw"].iloc[len(df) - 1]) if "price_raw" in df.columns else None

    @staticmethod
    def _add_tradeable(df):
        """
        See @_add_availability
        """
        return bool(df["tradeable"].iloc[len(df) - 1])

    def _add_date(self, df):
        if self.date_col_name in df.columns:
            return df[self.date_col_name].iloc[len(df) - 1]
        return None

    @staticmethod
    def _add_sentiment_data_availability(df):
        if "sentiment_data_available" in df.columns:
            return bool(df["sentiment_data_available"].iloc[len(df) - 1])
        return None

    def sliced_to_sequence_obj(self):
        """
        Transforms each slice into a Sequence object.
        """

        for slice in self._sliced:
            metadata = Metadata(
                ticker_name=self.ticker_name,
                available=self._add_availability(slice),
                tradeable=self._add_tradeable(slice),
                price=self._add_price(slice),
                price_raw=self._add_raw_price(slice),
                sentiment_data_available=self._add_sentiment_data_availability(slice),
                date=self._add_date(slice),
            )

            data = Data(df=slice)

            self._sequences.append(Sequence(metadata=metadata, data=data))

        return self._sequences

    def filter_availability(self) -> list:
        """
        Filters a sliced df based on whether the last element from this df is available or not.
        """
        if not self.include_available_days_only:
            return self._sequences

        self._sequences = [seq for seq in self._sequences if seq.metadata.available is True]
        return self._sequences

    def handle_column_order(self):
        """
        Reorders the columns of a sequence. The last_column will be the last column. (captain obvious)
        """
        if self.price_column is None or not self._sequences:  # If sequences is empty
            return self._sequences

        cols = self._sequences[0].data.df.columns.tolist()
        cols.remove(self.price_column)
        cols += [self.price_column]

        for seq in self._sequences:
            seq.data.df = seq.data.df[cols]
        return self._sequences

    def exclude_columns(self):
        """
        Drops columns from a sequence.
        """
        if not self.exclude_cols_from_sequence:
            return self._sequences

        for seq in self._sequences:
            seq.data.df = seq.data.df.drop(columns=self.exclude_cols_from_sequence)
        return self._sequences

    def convert_to_sequences(self):
        temp = Sequences()
        temp.lst = self._sequences
        self._sequences = temp

    def make_sequence(self):
        self.slice_sequences()
        self.sliced_to_sequence_obj()
        self.filter_availability()
        self.handle_column_order()
        self.exclude_columns()
        self.convert_to_sequences()
        return self._sequences

    @staticmethod
    def _reset_columns(df):
        df.columns = pd.RangeIndex(df.columns.size)
        return df

    def add_array_sequences(self):
        for seq in self._sequences:
            arr = seq.data.df.copy()
            arr = arr.transpose()
            arr = self._reset_columns(arr)
            seq.data.arr = arr
        return self._sequences

    def get_sequences(self):
        return self._sequences

    def add_flat_sequences(self):
        """
        Flattens a df. Example: If we have 10 rows and 5 columns we will have 1 row and 50 columns afterwards. Each
        value that gets transformed into a new column will have a prefix with an integer indicating the index in the
        original df it was located at.
        """
        # TODO Might be faster to use numpy reshape here - but we would have to deal with the column names then
        for seq in self._sequences:
            seq.data.flat = seq.data.df.unstack().to_frame().T
        return self._sequences

    def cleanup(self):
        for seq in self._sequences:
            seq.df = None
