import pickle as pkl
import warnings
from random import shuffle

import paths
from learning.agent import Agent
from learning.env import StockEnv


def main(input_path, continue_training=False, eval=False, model_path=None, eval_out_path="eval.pkl"):
    if eval:
        warnings.warn("Eval is active.")

        if model_path is None or eval_out_path is None:
            raise ValueError("Eval is activated but no model or eval out path is specified.")

        eval_out_path = model_path / eval_out_path

    with open(input_path, "rb") as f:
        data = pkl.load(f)

    env = StockEnv()
    state_size = data[0]["data"].shape[1] - 3  # -1 because we remove the "price", "tradeable" and "date" column
    agent = Agent(state_size=state_size, action_size=3, memory_len=1000, eval=eval)

    n_episodes = 3
    batch_size = 32

    if eval:
        agent.load(model_path)
        n_episodes = 1
        batch_size = 0
    if continue_training:
        agent.load(model_path)

    for i, grp in enumerate(data):

        print(f"{i + 1}/{len(data)} - Processing ticker: {grp['ticker']}")

        df = grp["data"].drop(columns=["price", "tradeable", "date"])

        if eval:
            actions = []
            actions_outputs = []

        for e in range(n_episodes):

            print(f"Episode {e + 1}/{n_episodes}")

            state = env.reset(df)
            done = False

            while not done:

                action, action_output = agent.act(state)

                if eval:
                    actions.append(action)
                    actions_outputs.append(action_output)

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
            grp["data"]["actions"] = actions
            grp["data"]["actions_outputs"] = actions_outputs

    if eval:
        with open(eval_out_path, "wb") as f:
            pkl.dump(data, f)
        return data

    else:
        agent.save(input_path.parent.name, paths.models_path)


if __name__ == "__main__":
    main(paths.d_path(12) / "timeseries.pkl",
         continue_training=False,
         eval=False,
         model_path=paths.models_path / "12-10_22---22_02-21"
         )
