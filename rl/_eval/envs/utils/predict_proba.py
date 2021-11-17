import numpy as np

from stable_baselines3.common.policies import obs_as_tensor


def predict_proba(model, state):
    obs = obs_as_tensor(state, model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()
    probs_lst = probs_np.tolist()[0]

    action = int(np.argmax(probs_np))
    return action, probs_lst
