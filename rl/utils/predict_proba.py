import numpy as np
from stable_baselines3.common.policies import obs_as_tensor


def predict_proba(model, state):
    obs = obs_as_tensor(state, model.policy.device)

    # Add batch dimension
    if isinstance(obs, dict):
        for key, value in obs.items():
            obs[key] = value.unsqueeze(0)
    else:
        obs = obs.unsqueeze(0)

    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()
    probs_lst = probs_np.tolist()[0]

    action = int(np.argmax(probs_np))
    return action, probs_lst
