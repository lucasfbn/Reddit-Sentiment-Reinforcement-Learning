from stable_baselines3.common.callbacks import BaseCallback


class Callback(BaseCallback):

    def __init__(self, episodes_log_interval):
        super().__init__(verbose=1)

        self._episodes_log_interval = episodes_log_interval
        self._episode_counter = 0

    def per_step(self):
        pass

    def per_episode(self):
        pass

    def per_episode_interval(self):
        pass

    def _on_step(self) -> bool:
        self.per_step()

        if self.locals["dones"]:
            self.per_episode()
            self._episode_counter += 1

        if self._episode_counter == self._episodes_log_interval:
            self.per_episode_interval()
            self._episode_counter = 0

        return True
