import orjson
import pandas as pd

from preprocessing.sequence import Eval as SequenceEval
from preprocessing.sequence import Sequence, Metadata, Data
from preprocessing.ticker import Eval as TickerEval
from preprocessing.ticker import Ticker, Sequences


class Parser:

    def __init__(self, fp):
        self.fp = fp

    def _parse_sequences(self, dic_sequences):
        sequences = []

        for dic_seq in dic_sequences:
            seq = Sequence()
            seq.index = dic_seq["index"]

            seq.data = Data()

            seq.metadata = Metadata()
            seq.metadata.__dict__.update(dic_seq["metadata"])
            # seq.metadata.date = pd.Period(seq.metadata.date) # Performance

            seq.evl = SequenceEval()
            seq.evl.__dict__.update(dic_seq["evl"])

            sequences.append(seq)

        return sequences

    def _parse_single(self, dic):
        ticker = Ticker()
        ticker.name = dic["name"]
        ticker.exclude = dic["exclude"]

        evl = TickerEval()
        evl.__dict__.update(dic["evl"])
        ticker.evl = evl

        sequences = Sequences()
        sequences.lst = self._parse_sequences(dic["sequences"])
        ticker.sequences = sequences

        return ticker

    def parse(self):
        with open(self.fp, "rb") as f:
            data = orjson.loads(f.read())
        print(1)

        return [self._parse_single(dic) for dic in data]
