from stable_baselines3.common.callbacks import BaseCallback


class EpisodeEndCallback(BaseCallback):

    def __init__(self, verbose=1):
        super(EpisodeEndCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        iteration = self.locals["iteration"]
        n_steps = self.locals["n_steps"]

        reward = self.locals["rewards"]
        action = self.locals["actions"]
        done = self.locals["dones"]

        env = self.training_env.envs[0].env
        env.log()
        return True
