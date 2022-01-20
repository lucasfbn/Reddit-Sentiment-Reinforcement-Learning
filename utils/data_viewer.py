import wandb
from utils.wandb_utils import load_file
import pandas as pd

with wandb.init(project="Trendstuff", group="Data Viewer") as run:
    data = load_file(run_id="3924e7w4", fn="tracking.pkl", run=run)

    # import pickle as pkl
    #
    # with open("temp.pkl", "wb") as f:
    #     pkl.dump(data[:10], f)

    mean_reward = pd.concat(e.to_df(include_last=False) for e in data[:-30])

    print(len(mean_reward[mean_reward["episode_end"] == True]))

    print()
