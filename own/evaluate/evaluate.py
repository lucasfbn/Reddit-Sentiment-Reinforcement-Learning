from own.env import StockEnv
from own.agent import Agent
import paths
import pickle as pkl

env = StockEnv()
state_size = env.observation_space
action_size = env.action_space

agent = Agent(state_size, action_size, memory_len=1000, eval=True)
agent.load(paths.models_path / "test2.mdl")

with open(paths.data_path / "data_timeseries.pkl", "rb") as f:
    data = pkl.load(f)

for grp in data:

    print(f"Processing ticker: {grp['ticker']}")

    prices_raw = grp["data"]["price_raw"]
    df = grp["data"].drop(columns=["price_raw"])

    actions = []

    state = env.reset(df)
    done = False

    while not done:

        action = agent.act(state)

        actions.append(action)

        next_state, reward, done, _ = env.step(action)

        if done:
            break

        state = next_state

    grp["data"]["actions"] = actions + [-1]  # -1 since we do not have an action for the last entry

with open(paths.data_path / "evaluated.pkl", "wb") as f:
    pkl.dump(data, f)
