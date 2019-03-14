# Copyright 2016 Google Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""1-Step Q-Learning. Parallel gameplay, synchronous algorithm execution (like A2C)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import six
import time
from multiprocessing import Process, Queue, Pipe
import copy
from pprint import pprint


import deepmind_lab

""" Dependencies used for this:
    pip uninstall numpy
    pip install --no-cache-dir numpy==1.15.4
    pip install --upgrade tensorflow
    pip install --upgrade tensorflow-probability
    pip install wrapt"""
import tensorflow as tf
import trfl


  
# General Parameters

train_episodes = 5000          # max number of episodes to learn from
                                # This is now number of steps per agent essentially 
max_steps = 5               # max steps before reseting the agent
gamma = 0.8                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.005            # exponential decay rate for exploration prob

# Network parameters
kernel_size_1 = [8,8,3]
output_filters_conv1 = 32     
output_filters_conv2 = 64     
output_filters_conv3 = 64     
hidden_size = 512               # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

# Memory parameters
# memory_size = 10000            # memory capacity
num_envs = batch_size = 4                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

# Training parameters
n = 5 # n in n-step updating
entropy_reg_term = 1 #1000000. #regularization term for entropy
normalise_entropy = False # when true normalizes entropy to be in [-1, 0] to be more invariant to different size action spaces

#target QN
# update_target_every = 10
state_size = (80,80,3)
action_size = 6


def _action(*entries):
  return np.array(entries, dtype=np.intc)

class ActorCriticNetwork:
    def __init__(self, name):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            # self.inputs_ = tf.placeholder(tf.float32, [n, num_envs, state_size[0], state_size[1], state_size[2]], name='inputs')
            # self.inputs_flat = tf.reshape(self.inputs_, [n * num_envs, state_size[0], state_size[1], state_size[2]])

            self.inputs_ = tf.placeholder(tf.float32, [None, state_size[0], state_size[1], state_size[2]], name='inputs')
            # Actions for the QNetwork:
            # One-hot vector, with each action being as follows:
            # (look_left, look_right, strafe_left, strafe_right, forward, backward)           
            # These are mapped to the deepmind-lab (not one-hot) actions with the same names
            # defined in ACTIONS

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [n, num_envs], name='actions')
            
            # ReLU hidden layers
            self.conv1 = tf.contrib.layers.conv2d(self.inputs_, output_filters_conv1, kernel_size=8, stride=2)
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, output_filters_conv2, kernel_size=4, stride=2)
            self.conv3 = tf.contrib.layers.conv2d(self.conv2, output_filters_conv3, kernel_size=4, stride=1)
            
            self.fc1 = tf.contrib.layers.fully_connected( \
              tf.reshape(self.conv3, [-1, self.conv3.shape[1]*self.conv3.shape[2]*self.conv3.shape[3]]), \
              hidden_size)


            # Value function - Linear output layer
            self.value_output = tf.contrib.layers.fully_connected(self.fc1, 1, 
                                                            activation_fn=None)

            # Policy - softmax output layer
            self.policy_logits = tf.contrib.layers.fully_connected(self.fc1, action_size, activation_fn=None)
            self.policy_output = tf.contrib.layers.softmax(self.policy_logits)


            self.name = name

            self.rewards = tf.placeholder(tf.float32,[n, num_envs],name="rewards")
            self.discounts = tf.placeholder(tf.float32,[n, num_envs],name="discounts")
            self.initial_Rs = tf.placeholder(tf.float32, [num_envs], name="initial_Rs")

            # Used for trfl stuff
            self.value_output_unflat = tf.reshape(self.value_output, [n, num_envs])
            self.policy_logits_unflat = tf.reshape(self.policy_logits, [n, num_envs, -1])

            print("Network shapes:")
            print("actions_: ", self.actions_.shape)
            print("conv1: ", self.conv1.shape)
            print("conv2: ", self.conv2.shape)
            print("conv3: ", self.conv3.shape)
            print("fc1: ", self.fc1.shape)
            print("value_output: ", self.value_output.shape)
            print("policy_logits: ", self.policy_logits.shape)
            print("policy_output: ", self.policy_output.shape)

            print("policy_logits_unflat: ", self.policy_logits_unflat.shape)
            print("value_output_unflat: ", self.value_output_unflat.shape)

            #TRFL qlearning
            a2c_loss, extra = trfl.sequence_advantage_actor_critic_loss(
                policy_logits = self.policy_logits_unflat,
                baseline_values = self.value_output_unflat, 
                actions = self.actions_, 
                rewards = self.rewards,
                pcontinues = self.discounts, 
                bootstrap_value = self.initial_Rs,
                entropy_cost = entropy_reg_term,
                normalise_entropy = normalise_entropy)
            self.loss = tf.reduce_mean(a2c_loss)
            self.extra = extra
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            
    def get_action(self, sess, state, t_list, print_policies=False):
        """ 
        Returns the action chosen by the QNetwork. 
        Should be called by the mainA2C 
        """
        feed = {mainA2C.inputs_: np.reshape(state, [-1, state_size[0], state_size[1], state_size[2]])}
        policies, values = sess.run([mainA2C.policy_output, mainA2C.value_output], feed_dict=feed)
        if print_policies:
            pprint(zip(policies, t_list))
        actions = [np.random.choice(len(policy), p=policy) for policy in policies]
        print("actions chosen: ", actions)
        # print("actions: ", actions)
        return actions
    
    def get_value(self, sess, state):
        """ 
        Returns the value of a state by the QNetwork. 
        Should be called by the mainA2C 
        """
        feed = {mainA2C.inputs_: np.reshape(state, [-1, state_size[0], state_size[1], state_size[2]])}
        # feed = {mainA2C.inputs_: state.reshape((batch_size, state.shape[0], state.shape[1], state.shape[2]))}
        policies, values = sess.run([mainA2C.policy_output, mainA2C.value_output], feed_dict=feed)
        # print("values: ", values)
        return values

    def train_step(self, sess, states, actions, rewards, discounts, initial_Rs):
        """
        Runs a train step
        Returns the loss
        Should be called by mainA2C
        """

        loss, extra, opt = sess.run([self.loss, self.extra, self.opt], 
                    feed_dict={self.inputs_: np.reshape(states, [-1, 80, 80, 3]),
                                self.actions_: actions,
                                self.rewards: rewards,
                                self.discounts: discounts,
                                self.initial_Rs: initial_Rs})
        return loss, extra


def get_random_action():
    # return 4
    return random.randint(0,5)

    # DeepMind Lab defines takes actions as follows:
    # ACTIONS = {
    #   'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
    #   'look_right': _action(20, 0, 0, 0, 0, 0, 0),
    #   # 'look_up': _action(0, 10, 0, 0, 0, 0, 0),
    #   # 'look_down': _action(0, -10, 0, 0, 0, 0, 0),
    #   'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
    #   'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
    #   'forward': _action(0, 0, 0, 1, 0, 0, 0),
    #   'backward': _action(0, 0, 0, -1, 0, 0, 0),
    #   # 'fire': _action(0, 0, 0, 0, 1, 0, 0),
    #   # 'jump': _action(0, 0, 0, 0, 0, 1, 0),
    #   # 'crouch': _action(0, 0, 0, 0, 0, 0, 1)
    # }


def map_to_dmlab(action_index):
    
    DMLAB_ACTIONS = [_action(-20, 0, 0, 0, 0, 0, 0),
    _action(20, 0, 0, 0, 0, 0, 0),
    _action(0, 0, -1, 0, 0, 0, 0),
    _action(0, 0, 1, 0, 0, 0, 0),
    _action(0, 0, 0, 1, 0, 0, 0),
    _action(0, 0, 0, -1, 0, 0, 0)]

    return DMLAB_ACTIONS[action_index]

def index_to_english(action):
  english_names_of_actions = [
    'look_left', 'look_right', 'strafe_left', 'strafe_right', 'forward', 'backward'
  ]
  return english_names_of_actions[action]


def env_worker(child_conn, level, config):
    env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config)
    env.reset()
    print("Started another environment worker!")

    while True:
        # if child_conn.poll():
        # Note: using the above loops, without it blocks. Not sure which is fastest.
            action, t = child_conn.recv()
            package = env_step(env, action, t)
            child_conn.send(package)
 
def env_step(env, action, t, num_repeats=60):

    # print(index_to_english(action))
    english_action = index_to_english(action)
    action = map_to_dmlab(action)
    reward = 0
    count = 0
    reset = False
    
    while count < num_repeats:

      if not env.is_running():
        env.reset()

      reward = env.step(action)
    
      if reward != 0:
        break 
      
      count +=1
    if reward > 0:
        print("Action: ", english_action, " REWARD: " + str(reward), "Steps taken: ", t)

    # Dealing w/ end of episode
    next_state = None
    episode_done = False
    # print("t: ", t, "max_steps: ", max_steps, "reward: ", reward)
    if reward > 0 or t == max_steps:
        if t == max_steps:
            reward = .0000001
        next_state = np.zeros(state_size)
        t = 0
        env.reset()
        next_state = env.observations()['RGB_INTERLEAVED']
        episode_done = True

    else:
        next_state = env.observations()['RGB_INTERLEAVED']
        t += 1

    return (next_state, reward, t, episode_done)

def reset_envs(env):
    env.reset()
    return env

def get_bootstrap(args, sess, mainA2C):
    # Getting R to use as initial condition. Essentially doing the whole target function thing. 
    state, action, next_state, reward, t, episode_done = args
    
    if reward == 0:
        next_state_data = np.expand_dims(np.array(next_state), axis=0)
        bootstrapped_R = np.max(mainA2C.get_value(sess, next_state_data)) # Shouldnt need to be a max
    else:
        bootstrapped_R = 0

    return bootstrapped_R

def deep_cast_to_nparray(bad_array):
    return np.array([np.array([np.array(a) for a in inner]) for inner in bad_array])

def train(level, config):
    # Now train with experiences

    # TODO: this is not used right now. Discount rewards
    # Initialization
    envs_list = [deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config)] * num_envs
    envs_list = map(reset_envs, envs_list)
    state_batch = map(lambda env: env.observations()['RGB_INTERLEAVED'], envs_list)
    next_state_batch = copy.deepcopy(state_batch)

    # Initalization of multiprocessing stuff
    pipes = [Pipe() for i in range(num_envs)]
    parent_conns, child_conns = zip(*pipes)

    processes = [Process(target=env_worker, 
          args=(child_conns[i],level, config)) 
          for i in range(num_envs)]

    for i in range(num_envs):
        processes[i].start()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # https://medium.com/@anthony_sarkis/tensorboard-quick-start-in-5-minutes-e3ec69f673af
        # train_writer = tf.summary.FileWriter( '/mnt/hgfs/ryanprinster/lab/tensorboard', sess.graph)

        step = 0 # Same step at every train iteration
        t_list = [0 for i in range(num_envs)]
            
        for ep in range(1, train_episodes):

            # n-steps
            n_steps_parallel = []
            for i in range(n):
                state_batch = next_state_batch # TODO: Namespace collision?

                step += 1
                print("step: ", step)
                print_policies = True
                # if step % 10 == 0:
                #     print_policies = True
                # GPU, PARALLEL
                action_list = mainA2C.get_action(sess, np.array(state_batch), t_list, print_policies)
                # action_list = map(apply_epsilon_greedy, zip(action_list, [step] * num_envs))

                # CPU, PARALLEL
                # Take action in environment, get new state and reward
                for i in range(num_envs):
                    package = (action_list[i], t_list[i])
                    parent_conns[i].send(package)

                nextstate_reward_t_episodedone_list = [parent_conns[i].recv() for i in range(num_envs)]
                next_state_batch, reward_list, t_list, episode_done_list = zip(*nextstate_reward_t_episodedone_list)

                env_tuples = zip(state_batch, action_list, next_state_batch, reward_list, t_list, episode_done_list)

                # Accumulate n-step experience
                n_steps_parallel.append(np.array(env_tuples))



            bootstrap_vals_list = [get_bootstrap(last_state, sess, mainA2C) for last_state in n_steps_parallel[0]]

            n_steps_parallel = [deep_cast_to_nparray(tup) for tup in np.moveaxis(n_steps_parallel, -1, 0)]
            state_list, action_list, _next_state_list, reward_list, _t_list, _episode_done_list = n_steps_parallel

            vec_f = np.vectorize(lambda x: 0 if x != 0 else gamma)
            pcontinues_list = np.reshape(np.array([vec_f(i) for i in reward_list]), [n,num_envs])


            print("action_list.shape: ", action_list)
            print("state_list.shape: ", state_list.shape)
            print("reward_list: ", reward_list)

            # Train step
            loss, extra = mainA2C.train_step(sess, state_list, action_list, reward_list, pcontinues_list, bootstrap_vals_list)
            
            print("total loss: ", loss)
            print("entropy: ", extra.entropy)
            print("entropy_loss: ", extra.entropy_loss)
            print("baseline_loss: ", extra.baseline_loss)
            print("policy_gradient_loss: ", extra.policy_gradient_loss)
            print("advantages: ", np.reshape(extra.advantages, [n, num_envs]))
            print("discounted_returns: ", np.reshape(extra.discounted_returns, [n, num_envs]))


        # print("Saving...")
        # saver.save(sess, '/mnt/hgfs/ryanprinster/lab/models/my_model', global_step=ep)

        # print("Resoring...")
        # saver.restore(sess, tf.train.latest_checkpoint('/mnt/hgfs/ryanprinster/lab/models/'))


tf.reset_default_graph()
mainA2C = ActorCriticNetwork(name='main_acn')


def run(length, width, height, fps, level, record, demo, demofiles, video):
  """Spins up an environment and runs the random agent."""
  config = {
      'fps': str(fps),
      'width': str(width),
      'height': str(height)
  }
  if record:
    config['record'] = record
  if demo:
    config['demo'] = demo
  if demofiles:
    config['demofiles'] = demofiles
  if video:
    config['video'] = video

  #Testing actions
  ACTIONS = {
      'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
      'look_right': _action(20, 0, 0, 0, 0, 0, 0),
      'look_up': _action(0, 10, 0, 0, 0, 0, 0),
      'look_down': _action(0, -10, 0, 0, 0, 0, 0),
      'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
      'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
      'forward': _action(0, 0, 0, 1, 0, 0, 0),
      'backward': _action(0, 0, 0, -1, 0, 0, 0),
      'fire': _action(0, 0, 0, 0, 1, 0, 0),
      'jump': _action(0, 0, 0, 0, 0, 1, 0),
      'crouch': _action(0, 0, 0, 0, 0, 0, 1)
  }



  train(level, config)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--length', type=int, default=10,
                      help='Number of steps to run the agent')
  parser.add_argument('--width', type=int, default=80,
                      help='Horizontal size of the observations')
  parser.add_argument('--height', type=int, default=80,
                      help='Vertical size of the observations')
  parser.add_argument('--fps', type=int, default=60,
                      help='Number of frames per second')
  parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
  parser.add_argument('--level_script', type=str,
                      default='tests/empty_room_test',
                      help='The environment level script to load')
  parser.add_argument('--record', type=str, default=None,
                      help='Record the run to a demo file')
  parser.add_argument('--demo', type=str, default=None,
                      help='Play back a recorded demo file')
  parser.add_argument('--demofiles', type=str, default=None,
                      help='Directory for demo files')
  parser.add_argument('--video', type=str, default=None,
                      help='Record the demo run as a video')

  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.length, args.width, args.height, args.fps, args.level_script,
      args.record, args.demo, args.demofiles, args.video)
