import paths
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from learning.evaluate.statistics import eval_statistics, stringify, plot_portfolio

price_col = "Close"
colors = {"hold": "y", "buy": "g", "sell": "r"}

with open(paths.data_path / "evaluated.pkl", "rb") as f:
    data = pkl.load(f)


def prepare(grp):
    df = grp["data"]
    df = df.drop(df.tail(1).index)
    df = df[[price_col, "actions"]]
    df["actions"] = df["actions"].replace({0: "hold", 1: "buy", 2: "sell"})
    return df


def generate_points(prices, actions):
    x = []
    y = []
    hue = []
    for i, (price, action) in enumerate(zip(prices, actions)):
        x.append(i)
        y.append(price)
        hue.append(action)
    return x, y, hue


def calculate_profit(prices, actions):
    # In case we didn't buy anything there is also not profit/loss
    if "buy" not in actions:
        return "-", 0

    profit = 1
    inventory = []

    for price, action in zip(prices, actions):
        if action == "hold":
            continue
        elif action == "buy":
            inventory.append(price)
        elif action == "sell":
            if inventory:
                for inv in inventory:

                    if inv == 0.0:
                        inv += 0.01

                    profit *= price / inv
                inventory = []
            else:
                continue

    return round(profit, 2), len(inventory)


statistics = {"profits": [], "positions": []}
max_stop = -1

if max_stop == -1:
    max_stop = len(data)

with PdfPages("eval.pdf") as pdf:
    for stop, grp in enumerate(data):

        plt.clf()

        print(f"Processing {stop}/{max_stop}")

        if stop == max_stop:
            break

        df = prepare(grp)

        prices = df[price_col].values.tolist()
        actions = df["actions"].values.tolist()

        x, y, hue = generate_points(prices, actions)

        sns.pointplot(x=x, y=y, hue=hue, palette=colors, linestyles="")
        sns.lineplot(data=df[price_col], color="black")

        profit, positions = calculate_profit(prices, actions)

        statistics["profits"].append(profit)
        statistics["positions"].append(positions)

        if positions == 0:
            plt.title(f"Ticker: {grp['ticker']}, Profit: {profit}")
        else:
            plt.title(f"Ticker: {grp['ticker']}, Profit: {profit}, OPEN POSITIONS! ({positions})")

        plt.plot()
        pdf.savefig()

    # Append a last statistics page to pdf
    plt.clf()
    statistics = eval_statistics(statistics)
    statistics_str = stringify(statistics)

    statistics_page = plt.figure()
    statistics_page.clf()
    statistics_page.text(0.5, 0.25, statistics_str, size=12, ha="center")
    pdf.savefig()

    # Append portfolio curve
    plt.clf()
    x, y = plot_portfolio(statistics)
    sns.lineplot(x=x, y=y, color="black")
    plt.plot()
    pdf.savefig()
