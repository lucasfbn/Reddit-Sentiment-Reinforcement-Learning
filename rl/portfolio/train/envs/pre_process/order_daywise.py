import itertools


def order_sequences_daywise(ticker):
    all_sequences = [list(t.sequences) for t in ticker]
    all_sequences = itertools.chain(*all_sequences)
    buys = [seq for seq in all_sequences if seq.evl.action == 1]
    return sorted(buys, key=lambda seq: (seq.metadata.date, 1 - seq.evl.buy_proba))
