from __future__ import absolute_import, division, print_function

import base64
# import IPython
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
# import PIL.Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()


class Agent:

    def __init__(self, env):
        self.env = env
        self.agent = None

        self._model = None
        self._optimizer = None

    def initialize_cnn_model(self):
        layers = []
        layers.append(Conv1D(filters=32, kernel_size=3, activation='relu',
                             input_shape=(7, 9)))
        # layers.append(MaxPooling1D(pool_size=2))
        layers.append(Flatten())
        layers.append(Dense(50, activation='relu'))
        layers.append(Dense(3, activation=None))

        self._model = sequential.Sequential(layers)
        self._optimizer = tf.keras.optimizers.Adam()

    def initialize_nn_model(self):
        layers = []
        layers.append(Dense(units=42, activation="relu", input_shape=(72,)))
        layers.append(Dense(3, activation=None))

        self._model = sequential.Sequential(layers)
        self._optimizer = tf.keras.optimizers.Adam()

    def initialize_agent(self):
        self.agent = dqn_agent.DqnAgent(self.env.time_step_spec(),
                                        self.env.action_spec(),
                                        q_network=self._model,
                                        optimizer=self._optimizer,
                                        td_errors_loss_fn=common.element_wise_squared_loss,
                                        train_step_counter=tf.Variable(0))

        self.agent.initialize()

    def get_agent(self):
        return self.agent


def game_agent(env, num_actions):
    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # it's output.
    dense_layers = [dense_layer(num_units) for num_units in (100, 50)]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()
    return agent
