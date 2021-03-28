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

from learning_tf.env import TradingEnv, EnvCNN
import paths
import pickle as pkl

with open(paths.datasets_data_path / "_13" / "timeseries.pkl", "rb") as f:
    data = pkl.load(f)

env = tf_py_environment.TFPyEnvironment(EnvCNN(data))

# q_net = q_network.QNetwork(input_tensor_spec=env.observation_spec(),
#                            action_spec=env.action_spec(),
#                            conv_layer_params=([(64, 3, 1)]))

layers = []
layers.append(Conv1D(filters=32, kernel_size=7, activation='relu',
                     input_shape=(7, 9)))
# layers.append(MaxPooling1D(pool_size=2))
layers.append(Flatten())
layers.append(Dense(50, activation='relu'))
layers.append(Dense(3, activation=None))

q_net = sequential.Sequential(layers)

optimizer = tf.keras.optimizers.Adam()

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(env.time_step_spec(),
                           env.action_spec(),
                           q_network=q_net,
                           optimizer=optimizer,
                           td_errors_loss_fn=common.element_wise_squared_loss,
                           train_step_counter=train_step_counter)

agent.initialize()

# How long should training run?
num_iterations = 1000
# How many initial random steps, before training start, to
# collect initial data.
initial_collect_steps = 10
# How many steps should we run each iteration to collect
# data from.
collect_steps_per_iteration = 50
# How much data should we store for training examples.
replay_buffer_max_length = 10000

batch_size = 64
# learning_rate = 1e-4
# How often should the program provide an update.
log_interval = 10

# How many episodes should the program use for each evaluation.
num_eval_episodes = 5
# How often should an evaluation occur.
eval_interval = 100

random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
                                                env.action_spec())


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        print(_)

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            if time_step.is_last():
                print("isch last")
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=env.batch_size,
    max_length=replay_buffer_max_length)


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


collect_data(env, random_policy, replay_buffer, initial_collect_steps)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)
iterator = iter(dataset)

agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    collect_data(env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()
    print(step)

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
