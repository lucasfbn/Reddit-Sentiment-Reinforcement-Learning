from learning.env import StockEnv
from learning.agent import Agent
import paths
import pickle as pkl
from random import shuffle

with open(paths.data_path / "data_timeseries.pkl", "rb") as f:
    data = pkl.load(f)
shuffle(data)

env = StockEnv()

state_size = data[0]["data"].shape[1] - 2  # -1 because we remove the "Close" and "tradeable" column
agent = Agent(state_size=state_size, action_size=3, memory_len=1000)

n_episodes = 3
batch_size = 32

for i, grp in enumerate(data):

    print(f"{i}/{len(data)} - Processing ticker: {grp['ticker']}")

    df = grp["data"].drop(columns=["Close", "tradeable"])

    for e in range(n_episodes):

        print(f"Episode {e}/{n_episodes}")

        state = env.reset(df)
        done = False

        while not done:

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            if done:
                break

            agent.remember(state, action, reward, next_state, done)

            state = next_state

        if len(agent.memory) > batch_size:
            # Note that agent.memory is a queue and we do not delete elements when replaying. Therefore, yes, we will
            # replay on the first loop when agent.memory == batch_size BUT we do not delete the content of the queue.
            # So it grows and eventually will throw out "old" state/action pairs.
            agent.replay(batch_size)

agent.save(paths.models_path)