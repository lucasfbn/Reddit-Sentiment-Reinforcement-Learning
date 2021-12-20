def merge_ticker(data_ticker, evl_ticker):
    assert len(data_ticker) == len(evl_ticker)

    for d_ticker, e_ticker in zip(data_ticker, evl_ticker):

        for d_seq, e_seq in zip(list(d_ticker.sequences), list(e_ticker.sequences)):
            e_seq.data = d_seq.data
            e_seq.metadata.ticker_name = d_seq.metadata.ticker_name

    return evl_ticker
