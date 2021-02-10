import warnings

from learning.env import StockEnv
from learning.agent import Agent
import paths
import pickle as pkl
from random import shuffle


def main(input_path, eval=False, model_path=None, eval_out_path=None):
    if eval:
        warnings.warn("Eval is active.")

        if model_path is None or eval_out_path is None:
            raise ValueError("Eval is activated but no model or eval out path is specified.")

        eval_out_path = model_path / eval_out_path

    with open(input_path, "rb") as f:
        data = pkl.load(f)
    shuffle(data)

    env = StockEnv()
    state_size = data[0]["data"].shape[1] - 2  # -1 because we remove the "Close" and "tradeable" column
    agent = Agent(state_size=state_size, action_size=3, memory_len=1000)

    n_episodes = 3
    batch_size = 32

    if eval:
        agent.load(model_path)
        n_episodes = 1
        batch_size = 0

    for i, grp in enumerate(data):

        print(f"{i+1}/{len(data)} - Processing ticker: {grp['ticker']}")

        df = grp["data"].drop(columns=["Close", "tradeable"])

        if eval:
            actions = []

        for e in range(n_episodes):

            print(f"Episode {e+1}/{n_episodes}")

            state = env.reset(df)
            done = False

            while not done:

                action = agent.act(state)

                if eval:
                    actions.append(action)

                next_state, reward, done, _ = env.step(action)

                if done:
                    break

                if not eval:
                    agent.remember(state, action, reward, next_state, done)

                state = next_state

            if not eval and len(agent.memory) > batch_size:
                # Note that agent.memory is a queue and we do not delete elements when replaying. Therefore, yes, we will
                # replay on the first loop when agent.memory == batch_size BUT we do not delete the content of the queue.
                # So it grows and eventually will throw out "old" state/action pairs.
                agent.replay(batch_size)

        if eval:
            grp["data"]["actions"] = actions + [-1]  # -1 since we do not have an action for the last entry

    if eval:
        with open(eval_out_path, "wb") as f:
            pkl.dump(data, f)
    else:
        agent.save(paths.models_path)


if __name__ == "__main__":
    main(paths.train_path / "data_timeseries.pkl",
         eval=True, model_path=paths.models_path / "18_44---08_02-21.mdl",
         eval_out_path="eval.pkl")
