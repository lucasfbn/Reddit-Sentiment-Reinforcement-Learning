import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
pd.options.mode.chained_assignment = None

def preprocess(df):
    df = df.drop(columns=['Unnamed: 0', 'Run Id'])
    df = df.rename({"Run Time (UTC)": "time"}, axis=1)

    new_cols = []
    for col in df.columns:
        col = col.lower()
        col = col.replace(" #", "")
        col = col.strip()
        col = col.replace(" ", "_")
        new_cols.append(col)

    df.columns = new_cols
    df["time"] = pd.to_datetime(df["time"], format="%d-%m-%Y %H:%M")
    df["date_day"] = pd.to_datetime(df['time']).dt.to_period('D')
    return df


def grp_by(df, ticker):
    grp = df.groupby(["ticker_symbol"])
    grp = grp.get_group(ticker)
    grp = grp.sort_values(by=["time"])
    grp = grp.groupby(["date_day"]).agg("sum").reset_index()
    return grp


def plt_several_splitted(df, cols):
    assert len(cols) > 1

    fig, axs = plt.subplots(nrows=len(cols), ncols=1)

    for i, col in enumerate(cols):
        print(i)
        s = sns.lineplot(x=ticker.index, y=ticker[col], ax=axs[i])
        s.set_xticklabels(ticker["date_day"].to_list(), rotation=30)

    plt.show()


def _normalize(df):
    normalizeable_cols = list(df.select_dtypes(include=[np.int64]).columns)

    for norm_col in normalizeable_cols:
        df[norm_col] = (df[norm_col] - df[norm_col].min()) / (df[norm_col].max() - df[norm_col].min())

    return df


def plt_several(ticker, cols, normalize=True):
    ticker = ticker[cols + ["date_day"]]

    if normalize:
        ticker = _normalize(ticker)

    melted = pd.melt(ticker, ['date_day'])
    melted["date_day_str"] = melted["date_day"].astype(str)

    s = sns.lineplot(x="date_day_str", y='value', hue='variable',
                     data=melted)
    s.set_xticklabels(melted["date_day_str"].to_list(), rotation=30)
    plt.show()


df = pd.read_csv("report_de.csv", sep=";")
df = preprocess(df)
ticker = grp_by(df, "ABML")

# print()
# plt_several(ticker, ["total_hype_level", "current_hype_level"])
