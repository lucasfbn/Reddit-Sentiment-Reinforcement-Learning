import pickle as pkl

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import paths
from evaluate.statistics import eval_statistics, stringify, plot_portfolio


class EvaluateTrades:
    price_col = "Close"
    colors = {"hold": "y", "buy": "g", "sell": "r"}

    def __init__(self, data, eval_out_path):
        self.data = data
        self.eval_out_path = eval_out_path

        self.statistics = {"profits": [], "positions": []}
        self.statistics_str = None

        self.prepare()
        self.calculate_profit()

    def prepare(self):
        for grp in self.data:
            df = grp["data"]
            df = df.drop(df.tail(1).index)  # since we do not have an action for the last entry
            df = df[[self.price_col, "actions", "tradeable"]]
            df["actions"] = df["actions"].replace({0: "hold", 1: "buy", 2: "sell"})
            grp["data"] = df

    def calculate_profit(self):

        for grp in self.data:
            df = grp["data"]

            prices = df[self.price_col].values.tolist()
            actions = df["actions"].values.tolist()
            tradeable = df["tradeable"].values.tolist()

            # In case we didn't buy anything there is also not profit/loss
            if "buy" not in actions:
                grp["profit"], grp["inventory"] = "-", 0
                self.statistics["profits"].append(grp["profit"])
                self.statistics["positions"].append(grp["inventory"])
                continue

            profit = 1
            inventory = []

            for price, action, trade_possible in zip(prices, actions, tradeable):
                if action == "hold":
                    continue
                elif action == "buy" and trade_possible:
                    inventory.append(price)
                elif action == "sell" and trade_possible:
                    if inventory:
                        for inv in inventory:

                            if inv == 0.0:
                                inv += 0.01

                            profit *= price / inv
                        inventory = []
                    else:
                        continue

            grp["profit"] = round(profit, 2)
            grp["inventory"] = len(inventory)
            self.statistics["profits"].append(grp["profit"])
            self.statistics["positions"].append(grp["inventory"])

    def overall_statistics(self):
        self.statistics = eval_statistics(self.statistics)
        self.statistics_str = stringify(self.statistics)
        return self.statistics

    def _generate_points(self, df):

        prices = df[self.price_col].values.tolist()
        actions = df["actions"].values.tolist()

        x = []
        y = []
        hue = []
        for i, (price, action) in enumerate(zip(prices, actions)):
            x.append(i)
            y.append(price)
            hue.append(action)
        return x, y, hue

    def plot(self, max_stop=-1, plot_statistics=True, plot_portfolio_curve=True):

        if max_stop == -1:
            max_stop = len(self.data)

        with PdfPages(self.eval_out_path) as pdf:

            if plot_statistics:
                plt.clf()
                statistics_page = plt.figure()
                statistics_page.clf()
                statistics_page.text(0.5, 0.25, self.statistics_str, size=12, ha="center")
                pdf.savefig()

            if plot_portfolio_curve:
                plt.clf()
                x, y = plot_portfolio(self.statistics)
                sns.lineplot(x=x, y=y, color="black")
                plt.plot()
                pdf.savefig()

            for stop, grp in enumerate(self.data):

                plt.clf()

                print(f"Processing {stop}/{max_stop}")

                if stop == max_stop:
                    break

                df = grp["data"]

                x, y, hue = self._generate_points(df)

                sns.pointplot(x=x, y=y, hue=hue, palette=self.colors, linestyles="")
                sns.lineplot(data=df[self.price_col], color="black")

                if grp['inventory'] == 0:
                    plt.title(f"Ticker: {grp['ticker']}, Profit: {grp['profit']}")
                else:
                    plt.title(f"Ticker: {grp['ticker']}, Profit: {grp['profit']}, OPEN POSITIONS! ({grp['inventory']})")

                plt.plot()
                pdf.savefig()

# with open(paths.models_path / "18_44---08_02-21.mdl" / "eval.pkl", "rb") as f:
#     data = pkl.load(f)
#
# evaluate = EvaluateTrades(data, paths.models_path / "18_44---08_02-21.mdl" / "eval.pdf")
# print(evaluate.overall_statistics())
# evaluate.plot()
# print()
