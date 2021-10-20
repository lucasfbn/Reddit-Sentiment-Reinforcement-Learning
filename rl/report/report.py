import gc
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from rl.envs.simple_trading import SimpleTradingEnv
from utils.pipeline_utils import task, par_map, initialize


class Plot:

    def __init__(self, ticker):
        self.ticker_name = ticker.name
        self.sequences = ticker.sequences

        self.map_actions()

    def map_actions(self):
        map_ = {0: "hold", 1: "buy", 2: "sell"}
        for seq in self.sequences:
            seq.action = map_[seq.action]

    def get_points(self):

        x, y, actions = [], [], []

        for i, seq in enumerate(self.sequences):
            x.append(i)
            y.append(seq.price)
            actions.append(seq.action)
        return x, y, actions

    def get_profit(self):
        ste = SimpleTradingEnv(self.ticker_name)

        profit = 1
        for seq in self.sequences:
            price = seq.price

            # Hold
            if seq.action == "hold":
                profit = ste.hold(profit, price)

            # Buy
            if seq.action == "buy":
                profit = ste.buy(profit, price)

            # Sell
            if seq.action == "sell":
                profit = ste.sell(profit, price)

        open_positions = True if len(ste.inventory) > 0 else False

        return profit, open_positions

    def get_plot(self):

        fig = plt.figure()

        x, y, actions = self.get_points()
        profit, open_positions = self.get_profit()

        sns.pointplot(x=x, y=y, hue=actions, palette={"hold": "y", "buy": "g", "sell": "r"}, linestyles="")
        sns.lineplot(x=x, y=y)
        plt.ylabel("Price (relative)")
        plt.xlabel("Day")
        plt.title(f"Ticker: {self.ticker_name}, Profit: {profit}, Open Positions: {str(open_positions)}")
        plt.plot()

        return fig


def make_pdf(data, path, fn="report.pdf"):
    ### CAUTION ###
    ### If there is a memory leak, delete the pycache (in the same folder as the current file)!!!
    ### CAUTION ###

    initialize()

    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)

    # Task for parallelization of img generation
    @task
    def make_imgs_(ticker):
        plot = Plot(ticker)
        fig_ = plot.get_plot()
        fig_.savefig(temp_dir_path / (ticker.name + ".jpg"), dpi=175)
        fig_.clear()
        plt.close(fig_)
        gc.collect()

    # Multiprocess above-mentioned task
    par_map(make_imgs_, data).run()

    # Make pdf from imgs
    def make_pdf_():
        imgs_path = [f for f in os.listdir(temp_dir_path) if f.endswith(".jpg")]
        imgs_loaded = [Image.open(temp_dir_path / img) for img in imgs_path]
        imgs_loaded[0].save(Path(path) / fn, "PDF", resolution=100.0, save_all=True, append_images=imgs_loaded[1:])

    make_pdf_()

    # Remove temp dir
    shutil.rmtree(temp_dir_path)
