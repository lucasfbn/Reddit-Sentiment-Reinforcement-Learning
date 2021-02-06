import numpy as np
import math
from sklearn.preprocessing import normalize


# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))

    return vec


# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# returns an an n-day state representation ending at time t

# Returned praktisch eine Zeitreihe bei deren letztes Element t ist und n Elemente davor.
def getState(data, t, n):
    # Basically the start point of the time series
    d = t - n + 1

    # If d > 0 than just return the sublist from d to t+1, else return pad the list with data[0]
    # Try: test = getState(list(range(100)), 5, 10) and look at "block"
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t0
    return np.array(block)
