import ray
from tensorforce import Agent


@ray.remote
def _predict_single(agent_path, env, x, get_probas):
    agent = Agent.load(directory=agent_path, format='numpy', tracking="all")

    for sequence in x.sequences:
        state = env.shape_state(sequence).df

        action = agent.act(state, independent=True)

        actions_proba = {"hold": None, "buy": None, "sell": None}
        if get_probas:
            arr = agent.tracked_tensors()["agent/policy/action_distribution/probabilities"]
            actions_proba = {"hold": arr[0], "buy": arr[1], "sell": arr[2]}

        sequence.add_eval(action, actions_proba)

    agent.close()

    return x


def predict_wrapper(agent_path, env, x, get_probas):
    ray.init()

    futures = [_predict_single.remote(agent_path=agent_path, env=env, x=x_, get_probas=get_probas) for x_ in x]
    predicted = ray.get(futures)

    ray.shutdown()
    return predicted
