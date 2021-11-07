import pickle as pkl

from utils.mlflow_api import load_file, init_mlflow

###
# Used for external use of a dataset
###

init_mlflow("Tests")
data = load_file(run_id="46345f27af9a42ebafc2590706469309", fn="ticker.pkl", experiment="Experimental_Datasets")

data_new = []

for ticker in data:
    data_new.append(
        dict(
            ticker=ticker.name,
            df=ticker.df
        )
    )

with open("exported.pkl", "wb") as f:
    pkl.dump(data_new, f)
