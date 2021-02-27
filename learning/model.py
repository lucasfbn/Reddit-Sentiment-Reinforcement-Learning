import warnings

from learning.agent import Agent
from learning.env import StockEnv


def deep_q_model(data, eval=False, model=None):
    if eval:
        warnings.warn("Eval is active.")

    env = StockEnv()
    state_size = data[0]["data"].shape[1] - 3  # -1 because we remove the "price", "tradeable" and "date" column
    agent = Agent(state_size=state_size, action_size=3, memory_len=1000, eval=eval)

    n_episodes = 3
    batch_size = 32

    if eval:
        agent.model = model
        n_episodes = 1
        batch_size = 0

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
        return data
    else:
        return agent.get_model()
