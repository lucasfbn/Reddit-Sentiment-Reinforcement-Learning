from refl.agent.agent import Agent
from refl.functions import *
import sys
import pickle as pkl
import paths
import random
from tensorflow.keras.models import load_model

with open(paths.data_path / "data_cleaned.pkl", "rb") as f:
    full_data = pkl.load(f)

window_size = 3

agent = Agent(window_size, True)
agent.model = load_model("models/" + "model2")

total_total_profit = 1

for i, data in enumerate(full_data):

    data["signals"] = []
    grps = data["data"]
    print(data["ticker"])
    grps = grps.values.tolist()
    l = len(grps)

    total_profit = 1
    agent.inventory = []

    state = getState(grps, 0, window_size + 1)

    # if data["ticker"] == "AUPH":
    #     print()
    # else:
    #     continue

    for t in range(l):
        action = agent.act(state)

        # sit
        next_state = getState(grps, t + 1, window_size + 1)
        reward = 0

        if action == 1:  # buy
            current_price = grps[t][len(grps[t]) - 1]
            agent.inventory.append(current_price)
            print(f"Buy: {current_price}")

        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            current_price = grps[t][len(grps)]
            reward = max(current_price - bought_price, 0)
            if bought_price == 0.0:
                continue
            current_profit =
            total_profit *= (current_price / bought_price)
            print(f"Sell: {current_price}, Profit: {current_price - bought_price}")

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
            break

    total_total_profit *= total_profit

print(total_total_profit)