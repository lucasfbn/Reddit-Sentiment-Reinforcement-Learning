class EpochCallback:

    def __init__(self):
        self._epochs = []
        self._epochs_total_reward = {}
        self._epoch_counter = -1

    def new_epoch(self):
        self._epoch_counter += 1
        self._epochs.append([])
        self._epochs_total_reward[self._epoch_counter] = 0

    def add_to_epoch(self, state, action, reward, next_state, done):
        self._epochs[self._epoch_counter].append((state, action, reward, next_state, done))
        self._epochs_total_reward[self._epoch_counter] += reward

    def _get_best_epoch(self):
        return max(self._epochs_total_reward, key=self._epochs_total_reward.get)

    def add_best_epoch_to_agent(self, agent):
        best_epoch = self._get_best_epoch()

        for entry in self._epochs[best_epoch]:
            agent.remember(entry[0], entry[1], entry[2], entry[3], entry[4])

# ec = EpochCallback()
#
# first_e = [
#     (0, 0, 1, 0, 0),
#     (0, 0, 1, 0, 0),
#     (0, 0, 1, 0, 0),
#     (0, 0, 1, 0, 0),
# ]
#
# second_e = [
#     (0, 0, 2, 0, 0),
#     (0, 0, 2, 0, 0),
#     (0, 0, 2, 0, 0),
#     (0, 0, 2, 0, 0),
# ]
#
# third_e = [
#     (0, 0, 1, 0, 0),
#     (0, 0, 2, 0, 0),
#     (0, 0, 3, 0, 0),
#     (0, 0, 4, 0, 0),
# ]
#
# ec.new_epoch()
# for e in first_e:
#     ec.add_to_epoch(e[0], e[1], e[2], e[3], e[4])
#
# ec.new_epoch()
# for e in second_e:
#     ec.add_to_epoch(e[0], e[1], e[2], e[3], e[4])
#
# ec.new_epoch()
# for e in third_e:
#     ec.add_to_epoch(e[0], e[1], e[2], e[3], e[4])
#
# print(ec._get_best_epoch())
