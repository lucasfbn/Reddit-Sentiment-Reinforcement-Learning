import warnings


def deep_q_model(data, agent, env, evaluate=False, model=None):
    if evaluate:
        warnings.warn("Eval is active.")

    n_episodes = 3
    batch_size = 32

    if evaluate:
        agent.model = model
        n_episodes = 1
        batch_size = 0

    for i, grp in enumerate(data):

        print(f"{i + 1}/{len(data)} - Processing ticker: {grp['ticker']}")
        x = grp["data"]

        if evaluate:
            actions = []
            actions_outputs = []

        for e in range(n_episodes):

            print(f"Episode {e + 1}/{n_episodes}")

            state = env.reset(x)
            done = False

            while not done:

                action, action_output = agent.act(state)

                if evaluate:
                    actions.append(action)
                    actions_outputs.append(action_output)

                next_state, reward, done, _ = env.step(action)

                if done:
                    break

                if not evaluate:
                    agent.remember(state, action, reward, next_state, done)

                state = next_state

            if not evaluate and len(agent.memory) > batch_size:
                # Note that agent.memory is a queue and we do not delete elements when replaying. Therefore, yes, we will
                # replay on the first loop when agent.memory == batch_size BUT we do not delete the content of the queue.
                # So it grows and eventually will throw out "old" state/action pairs.
                agent.replay(batch_size)

        if evaluate:
            grp["metadata"]["actions"] = actions
            grp["metadata"]["actions_outputs"] = actions_outputs

    if evaluate:
        return data
    else:
        return agent.get_model()
