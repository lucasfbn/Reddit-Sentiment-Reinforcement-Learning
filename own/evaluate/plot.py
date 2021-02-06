import paths
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

with open(paths.data_path / "evaluated.pkl", "rb") as f:
    data = pkl.load(f)

price_col = "Close"

data = data[0]
df = data["data"]
df = df.drop(df.tail(1).index)
df = df[[price_col, "actions"]]
df = df.replace({0: "hold", 1: "buy", 2: "sell"})

prices = df[price_col].values.tolist()
actions = df["actions"].values.tolist()

colors = {"hold": "y", "buy": "g", "sell": "r"}
x = []
y = []
hue = []
for i, (price, action) in enumerate(zip(prices, actions)):
    x.append(i)
    y.append(price)
    hue.append(action)

sns.pointplot(x=x, y=y, hue=hue, palette=colors, linestyles="")

sns.lineplot(data=df[price_col], color="black")

plt.title(f"Ticker: {data['ticker']}, Profit: {round(data['total_profit'], 2)}")
plt.show()
