from own.env import StockEnv
from own.agent import Agent
import paths
import pickle as pkl
from random import shuffle

env = StockEnv()
state_size = env.observation_space
action_size = env.action_space

agent = Agent(state_size, action_size, memory_len=1000)

with open(paths.data_path / "data_timeseries.pkl", "rb") as f:
    data = pkl.load(f)
shuffle(data)

n_episodes = 3
batch_size = 32

for grp in data:

    print(f"Processing ticker: {grp['ticker']}")

    df = grp["data"].drop(columns=["Close"])

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
