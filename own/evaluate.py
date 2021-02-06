from own.env import StockEnv
from own.agent import Agent
import paths
import pickle as pkl

env = StockEnv(eval=True)
state_size = env.observation_space
action_size = env.action_space

agent = Agent(state_size, action_size, memory_len=1000, eval=True)
agent.load(paths.models_path / "test2.mdl")

with open(paths.data_path / "data_timeseries.pkl", "rb") as f:
    data = pkl.load(f)

total_profit = 1

for grp in data:

    print(f"Processing ticker: {grp['ticker']}")

    prices_raw = grp["data"]["price_raw"]
    df = grp["data"].drop(columns=["price_raw"])

    state = env.reset(df, prices_raw)
    done = False

    while not done:

        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)

        if done:
            break

        state = next_state

    total_profit *= env.total_profit
    print(f"Current total profit: {total_profit}")

print(total_profit)
