from learning.env import StockEnv
from learning.agent import Agent
import paths
import pickle as pkl

env = StockEnv()


with open(paths.data_path / "data_timeseries.pkl", "rb") as f:
    data = pkl.load(f)

state_size = data[0]["data"].shape[1] - 2  # -1 because we remove the "Close" and "tradeable" column
agent = Agent(state_size=state_size, action_size=3, memory_len=1000, eval=True)
agent.load(paths.models_path / "14_42---08_02-21.mdl")


for grp in data:

    print(f"Processing ticker: {grp['ticker']}")

    df = grp["data"].drop(columns=["Close", "tradeable"])

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
