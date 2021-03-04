import random
from collections import deque

import numpy as np
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class Agent:

    def __init__(self, state_size, action_size, memory_len=1000, eval=False):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=memory_len)
        self.eval = eval

        # Discount future rewards
        self.gamma = 0.95

        # Exploration rate
        self.epsilon = 1.0
        # Over time, decay epsilon rate
        self.epsilon_decay = 0.999
        # To still explore we set a minimum epsilon which will not be influenced by the epsilon decay
        self.epsilon_min = 0.15

        self.model = None

    def act(self, state):
        if np.random.rand() <= self.epsilon and not self.eval:
            return random.randrange(self.action_size), -1
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]), np.max(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:

            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict([next_state])[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_model(self):
        return self.model


class NN_Agent(Agent):

    def build_model(self):
        model = Sequential()
        model.add(Dense(units=42, input_shape=(self.state_size,), activation="relu"))
        # model.add(Dense(units=21, activation="relu"))
        # model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam())
        self.model = model


class CNN_Agent(Agent):

    def __init__(self, state_size, action_size, feature_size, memory_len=1000, eval=False):
        super().__init__(state_size, action_size, memory_len=memory_len, eval=eval)
        self.feature_size = feature_size

    def build_model(self):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                         input_shape=(self.state_size, self.feature_size)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam())
        self.model = model
