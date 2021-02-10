import warnings

import numpy as np


def eval_statistics(statistics):
    print(statistics)
    profits = statistics["profits"]
    profits_np = np.array(profits)
    positions = statistics["positions"]
    positions = np.array(positions)

    statistics["total_evaluated"] = len(profits)

    if np.count_nonzero(np.where(profits_np == "-")) == 0:
        warnings.warn("Every single ticker was traded. This is probably not a good idea.")
        statistics["n_no_buys"] = 0
        profits_np = profits_np.astype(float)
    else:
        statistics["n_no_buys"] = np.count_nonzero(np.where(profits_np == "-")) / statistics["total_evaluated"]
        profits_np = profits_np[np.where(profits_np != "-")]
        profits_np = profits_np.astype(float)

    statistics["float_profits"] = profits_np.tolist()
    total_profit = 1
    for profit in statistics["float_profits"]:
        total_profit *= profit
    statistics["total_profit"] = total_profit

    statistics["n_profits"] = len(profits_np) / statistics["total_evaluated"]

    positive_profits = profits_np[np.where(profits_np > 1)]
    even_profits = profits_np[np.where(profits_np == 1)]
    negative_profits = profits_np[np.where(profits_np < 1)]

    statistics["n_positive_profits"] = np.count_nonzero(positive_profits) / len(profits_np)
    statistics["n_even_profits"] = np.count_nonzero(even_profits) / len(profits_np)
    statistics["n_negative_profits"] = np.count_nonzero(negative_profits) / len(profits_np)

    statistics["avg_profit"] = np.average(profits_np)
    statistics["avg_profit_negative"] = np.average(negative_profits)
    statistics["avg_profit_positive"] = np.average(positive_profits)

    statistics["median_profit"] = np.median(profits_np)
    statistics["median_profit_negative"] = np.median(negative_profits)
    statistics["median_profit_positive"] = np.median(positive_profits)

    statistics["n_positions"] = len(positions)
    statistics["n_open_positions"] = np.count_nonzero(np.where(positions != 0))
    statistics["rel_n_open_positions"] = statistics["n_open_positions"] / statistics["n_positions"]

    del statistics["profits"]
    del statistics["positions"]

    return statistics


def stringify(statistics):
    string = ""

    for key, value in statistics.items():

        if key != "float_profits":
            temp = f"{key}: {round(value, 2)}"
            string += temp + "\n"
    return string


def plot_portfolio(statistics, start=1):
    portfolio = [1]
    x = [0]

    i = 0
    for profit in statistics["float_profits"]:
        portfolio.append(portfolio[i] * profit)
        x.append(i + 1)
        i += 1

    return x, portfolio

# statistics = {}
# statistics["profits"] = [1.9, '-', 1.07, 34.11, '-']
# statistics["positions"] = [0, 0, 0, 0, 0]
# # #
#
# statistics = eval_statistics(statistics)
# x, y = plot_portfolio(statistics)
# print(x)
# print(y)
# print(stringify(statistics))
