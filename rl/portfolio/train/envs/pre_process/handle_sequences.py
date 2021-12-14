import itertools


def get_all_sequences(ticker):
    all_sequences = [list(t.sequences) for t in ticker]
    return itertools.chain(*all_sequences)


def remove_invalid_sequences(sequences):
    sequences = [seq for seq in sequences if seq.evl.reward_backtracked is not None]
    sequences = [seq for seq in sequences if seq.evl.action == 1]
    return sequences


def order_sequences_daywise(sequences):
    buys = [seq for seq in sequences if seq.evl.action == 1]
    return sorted(buys, key=lambda seq: (seq.metadata.date, 1 - seq.evl.buy_proba))
