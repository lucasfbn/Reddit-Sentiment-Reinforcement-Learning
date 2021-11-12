from stable_baselines3.common.callbacks import BaseCallback


class CB(BaseCallback):

    def __init__(self, verbose):
        super(CB, self).__init__(verbose)

    def _on_step(self) -> bool:
        env = self.training_env
        env = env.envs[0].env.log()
        return True
