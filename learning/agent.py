import datetime
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
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

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(units=64, input_shape=(self.state_size,), activation="relu"))
        model.add(Dense(units=48, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon and not self.eval:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0]), np.max(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:

            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict([np.array([next_state])])[0])

            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target

            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def save(self, path):
        now = datetime.datetime.now()
        now = now.strftime("%H_%M---%d_%m-%y")

        self.model.save(path / (now + ".mdl"))
