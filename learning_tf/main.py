from __future__ import absolute_import, division, print_function

import base64
# import IPython
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
# import PIL.Image
from tf_agents.specs import tensor_spec

from tqdm.auto import tqdm
from utils import log
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_episode_driver
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
from learning_tf.agent import Agent, game_agent


class Training:
    replay_buffer_max_length = 1000
    batch_size = 64
    log_interval = 200
    eval_interval = 1000

    def __init__(self, agent, env, n_training_episodes, n_eval_episodes, collect_steps_per_iteration,
                 initial_data_collect_runs):
        self.env = env
        self.agent = agent

        # How many episodes should the program use for each evaluation.
        self.n_eval_episodes = n_eval_episodes

        # How many episodes should the program train.
        self.n_training_episodes = n_training_episodes

        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.initial_data_collect_runs = initial_data_collect_runs * self.collect_steps_per_iteration

        self._replay_buffer = self._init_replay_buffer()  #
        self._driver_replay_buffer = []
        self._random_policy = self._init_random_policy()

    def _init_replay_buffer(self):
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.env.batch_size,
            max_length=self.replay_buffer_max_length)
        return replay_buffer

    def _init_random_policy(self):
        random_policy = random_tf_policy.RandomTFPolicy(self.env.time_step_spec(),
                                                        self.env.action_spec())
        return random_policy

    def compute_avg_return(self, policy):
        total_return = 0.0
        log.info("Calculating avg return...")
        for _ in tqdm(range(self.n_eval_episodes)):

            time_step = self.env.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = self.env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / self.n_eval_episodes
        return avg_return.numpy()[0]

    def _collect_step(self, policy):
        time_step = self.env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = self.env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        self._replay_buffer.add_batch(traj)

    def collect_data(self, policy, steps):
        for _ in range(steps):
            self._collect_step(policy=policy)

    def _pre_training(self):
        self.collect_data(self._random_policy, steps=self.initial_data_collect_runs)

    def run_driver(self):
        metric = py_metrics.AverageReturnMetric()
        replay_buffer = []
        observers = [self._driver_replay_buffer.append]
        driver = py_driver.PyDriver(env=self.env, policy=self.agent.collect_policy,
                                    max_steps=self.collect_steps_per_iteration,
                                    max_episodes=self.n_training_episodes, observers=observers)
        print("HIO")
        initial_time_step = self.env.reset()
        print(initial_time_step)
        final_time_step, _ = driver.run(initial_time_step)

        print('Replay Buffer:')
        for traj in self._driver_replay_buffer:
            print(traj)

        # print('Average Return: ', metric.result())

    def run(self):
        self.agent.train = common.function(self.agent.train)
        self.agent.train_step_counter.assign(0)

        # eval once before training
        self.collect_data(self._random_policy, steps=self.initial_data_collect_runs)

        dataset = self._replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(3)
        dataset_iter = iter(dataset)
        returns = [self.compute_avg_return(policy=self.agent.policy)]

        log.info("Training...")
        for _ in range(self.n_training_episodes):

            # Collect a few steps using collect_policy and save to the replay buffer.
            self.collect_data(policy=self.agent.collect_policy, steps=1)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, _ = next(dataset_iter)
            train_loss = self.agent.train(experience).loss

            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print(f'step = {step}: loss = {train_loss}')

            if step % self.eval_interval == 0:
                avg_return = self.compute_avg_return(policy=self.agent.policy)
                print(f'step = {step}: Average Return = {avg_return}')
                returns.append(avg_return)


with open(paths.datasets_data_path / "_13" / "timeseries.pkl", "rb") as f:
    data = pkl.load(f)

# env = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))
env = tf_py_environment.TFPyEnvironment(EnvCNN(data))

agent = Agent(env)
agent.initialize_model()
agent.initialize_agent()
agent = agent.get_agent()

# action_tensor_spec = tensor_spec.from_spec(env.action_spec())
# num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
# agent = game_agent(env, num_actions)

t = Training(env=env, agent=agent, n_training_episodes=10000, n_eval_episodes=10, collect_steps_per_iteration=1,
             initial_data_collect_runs=5000)
t.run()
