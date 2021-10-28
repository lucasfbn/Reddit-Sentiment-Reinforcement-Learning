import ray
from tensorforce import Agent


@ray.remote
def _predict_single(agent_path, env, x):
    agent = Agent.load(directory=agent_path, format='numpy', tracking="all")

    for sequence in x.sequences:
        state = env.shape_state(sequence).df

        action = agent.act(state, independent=True)

        arr = agent.tracked_tensors()["agent/policy/action_distribution/probabilities"]
        actions_proba = {"hold": arr[0], "buy": arr[1], "sell": arr[2]}

        sequence.add_eval(action, actions_proba)

    agent.close()

    return x


def predict_wrapper(agent_path, env, x):
    if not ray.is_initialized():
        ray.init()

    futures = [_predict_single.remote(agent_path=agent_path, env=env, x=x_) for x_ in x]
    predicted = ray.get(futures)
    return predicted
