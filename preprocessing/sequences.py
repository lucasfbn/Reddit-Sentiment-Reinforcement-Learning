import pandas as pd


class Sequence:

    def __init__(self, df: pd.DataFrame, sequence_len: int, include_available_days_only: bool,
                 exclude_cols_from_sequence: list = []):
        """
        Generates sequences from a given dataframe.

        Args:
            df: Input dataframe
            sequence_len: Length of the desired sequence
            include_available_days_only: Whether to filter sequences which were not available for trading
            exclude_cols_from_sequence: Cols that may be used during the generation of sequences but are not subject of
             the final sequences and can therefore be dropped
        """
        self.df = df
        self.sequence_len = sequence_len
        self.include_available_days_only = include_available_days_only
        self.exclude_cols_from_sequence = exclude_cols_from_sequence

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
            self._sequences.append(sliced)
        return self._sequences

    def filter_availability(self) -> list:
        """
        Filters a sliced df based on whether the last element from this df is available or not. We take the last element
        as a reference since this is the "current" element in this case - the elements prior to the last element are
        "past" elements. For instance, if we look at wednesday and have the data from monday and tuesday - it doesn't
        matter whether it was possible to trade on monday or tuesday, all that matters is that we can trade on
        wednesday.
        """
        if not self.include_available_days_only:
            return self._sequences

        valid_sequences = []
        for sequence in self._sequences:
            availability_of_last_sequence_element = sequence["available"].iloc[len(sequence) - 1]

            if availability_of_last_sequence_element == True:
                valid_sequences.append(sequence)

        self._sequences = valid_sequences
        return self._sequences

    def exclude_columns(self):
        new_sequences = []
        for seq in self._sequences:
            new_sequences.append(seq.drop(columns=self.exclude_cols_from_sequence))
        self._sequences = new_sequences
        return self._sequences

    def _make_sequence(self):
        self.slice_sequences()
        self.filter_availability()
        self.exclude_columns()

    def make_sequence(self):
        """
        Wrapper to run through all steps to create a sequence
        """
        raise NotImplementedError

    def to_list(self):
        return self._sequences


class FlatSequence(Sequence):

    def flatten(self):
        """
        Flattens a df. Example: If we have 10 rows and 5 columns we will have 1 row and 50 columns afterwards. Each
        value that gets transformed into a new column will have a prefix with an integer indicating the index in the
        original df it was located at.
        """
        # TODO Might be faster to use numpy reshape here - but we would have to deal with the column names then
        new_sequences = []
        for sequence in self._sequences:
            new_sequences.append(sequence.unstack().to_frame().sort_index(level=1).T)
        self._sequences = new_sequences
        return self._sequences

    def make_sequence(self):
        self._make_sequence()
        self.flatten()
        return self._sequences


class ArraySequence(Sequence):

    def make_sequence(self):
        self._make_sequence()
        return self._sequences
