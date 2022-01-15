import wandb
from utils.wandb_utils import load_file

with wandb.init(project="Trendstuff", group="Data Viewer") as run:
    data = load_file(run_id="", fn="", run=run)

    import pickle as pkl

    with open("temp.pkl", "wb") as f:
        pkl.dump(data[:10], f)

    print()
