import datetime

from tqdm import tqdm

import paths
from utils import log


class DeepQ:

    def __init__(self, data, agent, env):
        self.data = data
        self.agent = agent
        self.env = env

        self._evaluate = False

    def _generate_progressbar(self, current_episode, total_episodes):
        prgb = tqdm(self.data)
        prgb.set_description(f"Episode {current_episode}/{total_episodes}")
        return prgb

    def _episode(self, data_prg_iter, batch_size):

        episode_profit = 0

        for grp in data_prg_iter:

            actions = []
            actions_outputs = []

            x = grp["data"]
            state = self.env.reset(x)
            done = False

            while not done:

                action, action_output = self.agent.act(state)

                if self._evaluate:
                    actions.append(action)
                    actions_outputs.append(action_output)

                next_state, reward, done, _ = self.env.step(action)

                if done:
                    break

                if not self._evaluate:
                    self.agent.remember(state, action, reward, next_state, done)

                state = next_state

            episode_profit += self.env.total_profit

            if not self._evaluate and len(self.agent.memory) > batch_size:
                # Note that agent.memory is a queue and we do not delete elements when replaying. Therefore, yes, we will
                # replay on the first loop when agent.memory == batch_size BUT we do not delete the content of the queue.
                # So it grows and eventually will throw out "old" state/action pairs.
                self.agent.replay(batch_size)

            if self._evaluate:
                grp["metadata"]["actions"] = actions
                grp["metadata"]["actions_outputs"] = actions_outputs

        return episode_profit

    def train(self, n_episodes, batch_size):
        for e in range(1, n_episodes + 1):
            episode_profit = self._episode(data_prg_iter=self._generate_progressbar(e, n_episodes),
                                           batch_size=batch_size)
            log.info(f"Episode profit: {episode_profit}")

            if e % 10 == 0:
                self.agent.model.save(
                    paths.models_temp_path / f"temp_{datetime.datetime.now().strftime('%H-%M %d_%m-%y')}")

        return self.agent.get_model()

    def evaluate(self, model):
        self._evaluate = True
        self.agent.evaluate = True
        self.agent.model = model
        self._episode(data_prg_iter=self._generate_progressbar(1, 1),
                      batch_size=0)
        self.agent.evaluate = False
        self._evaluate = False
        return self.data
