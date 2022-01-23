import numpy as np
import pandas as pd


def rel_auc(series: pd.Series, quantiles=None, key=""):
    if quantiles is None:
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    len_ = len(series)

    arr = series.to_numpy()

    r = {f"auc_{key}": np.trapz(arr, dx=1) / len_}
    for q in quantiles:
        q_len = int(len_ * q)
        r[f"auc_{key}_{q}"] = np.trapz(arr[q_len:], dx=1) / len_

    return r
