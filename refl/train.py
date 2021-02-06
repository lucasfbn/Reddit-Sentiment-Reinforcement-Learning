from refl.agent.agent import Agent
from refl.functions import *
import sys
import pickle as pkl
import paths
import random

window_size = 5
episode_count = 2

agent = Agent(window_size)

with open(paths.data_path / "data_cleaned.pkl", "rb") as f:
    full_data = pkl.load(f)

random.shuffle(full_data)

batch_size = 2

for i, data in enumerate(full_data):

    print(f"Datapair {i}/{len(full_data)}")

    data = data["data"]
    data = data.values.tolist()
    l = len(data)

    # Basically training loop
    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))

        # Returns a list of stockprices within a certain window
        # Diese Funktion ändern um unsere Einflussvariablen zu nutzen
        state = getState(data, 0, window_size + 1)

        total_profit = 0
        agent.inventory = []

        for t in range(l):
            action = agent.act(state)

            # sit
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0

            if action == 1:  # buy
                current_price = data[t][len(data[t]) - 1]
                agent.inventory.append(current_price)
                print(f"Buy: {current_price}")

            elif action == 2 and len(agent.inventory) > 0:  # sell
                bought_price = agent.inventory.pop(0)
                current_price = data[t][len(data[t]) - 1]
                reward = max(current_price - bought_price, 0)
                total_profit += current_price - bought_price
                print(f"Sell: {current_price}, Profit: {current_price - bought_price}")



            # Wenn am Ende angekommen
            done = True if t == l - 1 else False

            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            # Merkt sich die States
            # (die Frage ist hier ob man sich die Hype Level merken sollte wenn keine Handelsdaten vorliegen)
            # State ist hier der previous state, next state der aktuelle
            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

            if done:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("--------------------------------")
                break



agent.model.save("models/model2")




















