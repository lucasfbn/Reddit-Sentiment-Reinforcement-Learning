import mlflow


class RewardCounter:

    def __init__(self):
        self.full = 0
        self.neg = 0
        self.even = 0
        self.pos = 0

    def add_reward(self, reward):
        self.full += reward

        if reward == 0.0:
            self.even += 1
        elif reward > 0.0:
            self.pos += 1
        elif reward < 0.0:
            self.neg += 1
        else:
            raise ValueError("Reward")

    def log(self, step):
        mlflow.log_metric("full run reward", self.full, step=step)
        mlflow.log_metric("n_even_rewards", self.even, step=step)
        mlflow.log_metric("n_pos_rewards", self.pos, step=step)
        mlflow.log_metric("n_neg_rewards", self.neg, step=step)
