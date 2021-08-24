from utils.mlflow_api import load_file
import pandas as pd
import pickle as pkl
import mlflow
import paths

mlflow.set_tracking_uri(paths.mlflow_path)
ticker = load_file(fn="eval.pkl", run_id="c85d728efd4242c796b917e20ca8ce47", experiment="Live")

dicts = []

for ticker in ticker:
    for seq in ticker.sequences:
        seq_dict = dict(
            ticker=ticker.name,
            price=seq.price_raw,
            date=seq.date,
            tradeable=seq.tradeable,
            action=seq.action,
            action_probas=seq.action_probas,
            hold=seq.action_probas["hold"],
            buy=seq.action_probas["buy"],
            sell=seq.action_probas["sell"],
        )
        dicts.append(seq_dict)

df = pd.DataFrame(dicts)

df.to_csv("temp.csv", sep=";")
