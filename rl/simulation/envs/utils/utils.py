import pandas as pd

from rl.simulation.envs.utils.operation import Operation


def ticker_list_to_df(ticker):
    dicts = []

    for ticker_id, ticker in enumerate(ticker):
        for seq_id, seq in enumerate(ticker.sequences):
            seq_dict = dict(
                ticker=ticker.name,
                price=seq.metadata.price_raw,
                date=seq.metadata.date,
                tradeable=seq.metadata.tradeable,
                action=seq.evl.action,
                action_probas=seq.evl.probas,
                # hold=seq.action_probas["hold"],
                # buy=seq.action_probas["buy"],
                # sell=seq.action_probas["sell"],
                ticker_id=ticker_id,
                seq_id=seq_id
            )
            dicts.append(seq_dict)

    return pd.DataFrame(dicts)


def order_day_wise(ticker, df):
    df = df.sort_values(by="date")
    grps = df.groupby(["date"])

    day_wise = {}

    for name, grp in grps:
        temp = []

        def to_operation(row):
            curr_ticker = ticker[row["ticker_id"]]
            curr_sequence = curr_ticker.sequences.lst[row["seq_id"]]
            temp.append(Operation(curr_ticker, curr_sequence))

        pd.DataFrame(grp).apply(to_operation, axis="columns")
        day_wise[name] = temp

    return day_wise
