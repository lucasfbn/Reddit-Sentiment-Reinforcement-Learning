import pandas as pd


class Sequence:

    def __init__(self, price, available, tradeable, df):
        self.df = df
        self.tradeable = tradeable
        self.available = available
        self.price = price

        self.action = None
        self.action_output = None

    def add_eval(self, action, action_output):
        """
        Used later on when the sequence will be evaluated.
        """
        self.action = action
        self.action_output = action_output

    def __len__(self):
        return len(self.df)


class SequenceGenerator:

    def __init__(self, df: pd.DataFrame, sequence_len: int, include_available_days_only: bool, price_column: str = None,
                 exclude_cols_from_sequence: list = []):
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

    @staticmethod
    def _add_tradeable(df):
        """
        See @_add_availability
        """
        return bool(df["tradeable"].iloc[len(df) - 1])

    def sliced_to_sequence_obj(self):
        """
        Transforms each slice into a Sequence object.
        """

        for slice in self._sliced:
            self._sequences.append(
                Sequence(
                    available=self._add_availability(slice),
                    tradeable=self._add_tradeable(slice),
                    price=self._add_price(slice),
                    df=slice
                ))
        return self._sequences

    def filter_availability(self) -> list:
        """
        Filters a sliced df based on whether the last element from this df is available or not.
        """
        if not self.include_available_days_only:
            return self._sequences

        self._sequences = [seq for seq in self._sequences if seq.available is True]
        return self._sequences

    def handle_column_order(self):
        """
        Reorders the columns of a sequence. The last_column will be the last column. (captain obvious)
        """
        if self.price_column is None or not self._sequences:  # If sequences is empty
            return self._sequences

        cols = self._sequences[0].df.columns.tolist()
        cols.remove(self.price_column)
        cols += [self.price_column]

        for seq in self._sequences:
            seq.df = seq.df[cols]
        return self._sequences

    def exclude_columns(self):
        """
        Drops columns from a sequence.
        """
        if not self.exclude_cols_from_sequence:
            return self._sequences

        for seq in self._sequences:
            seq.df = seq.df.drop(columns=self.exclude_cols_from_sequence)
        return self._sequences

    def _make_sequence(self):
        self.slice_sequences()
        self.sliced_to_sequence_obj()
        self.filter_availability()
        self.handle_column_order()
        self.exclude_columns()

    def make_sequence(self):
        """
        Wrapper to run through all steps to create a sequence
        """
        raise NotImplementedError

    def to_list(self):
        return self._sequences


class FlatSequenceGenerator(SequenceGenerator):

    def flatten(self):
        """
        Flattens a df. Example: If we have 10 rows and 5 columns we will have 1 row and 50 columns afterwards. Each
        value that gets transformed into a new column will have a prefix with an integer indicating the index in the
        original df it was located at.
        """
        # TODO Might be faster to use numpy reshape here - but we would have to deal with the column names then
        for sequence in self._sequences:
            sequence.df = sequence.df.unstack().to_frame().T
        return self._sequences

    def make_sequence(self):
        self._make_sequence()
        self.flatten()
        return self._sequences


class ArraySequenceGenerator(SequenceGenerator):

    def make_sequence(self):
        self._make_sequence()
        return self._sequences
