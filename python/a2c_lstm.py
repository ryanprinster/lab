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

""" 
Dependencies used for this:
pip uninstall numpy
pip install --no-cache-dir numpy==1.15.4
pip install --upgrade tensorflow
pip install --upgrade tensorflow-probability
pip install wrapt
"""
import tensorflow as tf
import trfl
  
# ______PARAMETERS______


# Network parameters
state_size = (80,80,3)
action_size = 6

kernel_size_1 = [8,8,3]
output_filters_conv1 = 32     
output_filters_conv2 = 64     
output_filters_conv3 = 64     
hidden_size = 512               # number of units in hidden layer
lstm_size = 256


# Training parameters
train_episodes = 5000#500          # max number of episodes to learn from
num_envs = 4                    # experience mini-batch size

# global learning_rate
# global max_steps
# global gamma
# global entropy_reg_term

learning_rate = 0.001          # learning rate
n = 20                          # n in n-step updating
max_steps = num_envs*train_episodes*n    # max steps before reseting the agent
gamma = 0.8                     # future reward discount
entropy_reg_term = 0.05           # regularization term for entropy
normalise_entropy = False       # when true normalizes entropy to be in [-1, 0] to be more invariant to different size action spaces


class ActorCriticNetwork:
    def __init__(self, name, num_envs=4, n=20):
        with tf.variable_scope(name):
            self.name = name

            # Input images
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size[0], state_size[1], state_size[2]], name='inputs')

            # One hot encode the actions:
            # [look_left, look_right, strafe_left, strafe_right, forward, backward]
            self.actions = tf.placeholder(tf.int32, [None, num_envs], name='actions')
            self.rewards = tf.placeholder(tf.float32, [None, num_envs], name='rewards')
            
            # Conv layers
            self.conv1 = tf.contrib.layers.conv2d(self.inputs_, output_filters_conv1, kernel_size=8, stride=2)
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, output_filters_conv2, kernel_size=4, stride=2)
            self.conv3 = tf.contrib.layers.conv2d(self.conv2, output_filters_conv3, kernel_size=4, stride=1)

            # Constructing input to AC network
            self.actions_input = tf.reshape(tf.one_hot(self.actions, action_size), [-1, action_size])
            self.rewards_input = tf.reshape(self.rewards, [-1, 1])
            self.vision_input = tf.reshape(self.conv3, [-1, self.conv3.shape[1]*self.conv3.shape[2]*self.conv3.shape[3]])

            self.ac_input = tf.concat([self.actions_input, self.rewards_input, self.vision_input], axis=1)

            # FC Layer
            self.fc1 = tf.contrib.layers.fully_connected(self.ac_input, hidden_size)

            # LSTM Layer
            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=False)
            self.lstm_hidden_state_input = tf.placeholder_with_default(
                                        self.lstm_cell.zero_state(batch_size=num_envs, dtype=tf.float32),
                                        [num_envs, hidden_size])
            # Should be lstm_size not hidden_size?


            self.lstm_input = tf.reshape(self.fc1, [-1, num_envs, hidden_size])

            # Dynamic RNN code - might not need to be dynamic
            self.lstm_output, self.lstm_hidden_state_output = tf.nn.dynamic_rnn(
                self.lstm_cell,
                self.lstm_input,
                initial_state=self.lstm_hidden_state_input,
                dtype=tf.float32,
                time_major=True,
                # parallel_iterations=num_envs, # Note: not sure what these do
                # swap_memory=True, # Note: not sure what these do
            )

            self.lstm_output_flat = tf.reshape(self.lstm_output, [-1, lstm_size])


            # Value function - Linear output layer
            self.value_output = tf.contrib.layers.fully_connected(self.lstm_output_flat, 1, 
                                                            activation_fn=None)

            # Policy - softmax output layer
            self.policy_logits = tf.contrib.layers.fully_connected(self.lstm_output_flat, action_size, activation_fn=None)
            self.policy_output = tf.contrib.layers.softmax(self.policy_logits)
            # Action sampling op
            self.action_output = tf.squeeze(tf.multinomial(logits=self.policy_logits,num_samples=1), axis=1)

            # Used for TRFL stuff
            self.value_output_unflat = tf.reshape(self.value_output, [n, num_envs])
            self.policy_logits_unflat = tf.reshape(self.policy_logits, [n, num_envs, -1])

            self.discounts = tf.placeholder(tf.float32,[n, num_envs],name="discounts")
            self.initial_Rs = tf.placeholder(tf.float32, [num_envs], name="initial_Rs")

            #TRFL loss
            a2c_loss, extra = trfl.sequence_advantage_actor_critic_loss(
                policy_logits = self.policy_logits_unflat,
                baseline_values = self.value_output_unflat, 
                actions = self.actions, 
                rewards = self.rewards,
                pcontinues = self.discounts, 
                bootstrap_value = self.initial_Rs,
                entropy_cost = entropy_reg_term,
                normalise_entropy = normalise_entropy)
            self.loss = tf.reduce_mean(a2c_loss)
            self.extra = extra
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            print("Network shapes:")
            print("inputs_: ", self.inputs_.shape)
            print("actions: ", self.actions.shape)
            print("conv1: ", self.conv1.shape)
            print("conv2: ", self.conv2.shape)
            print("conv3: ", self.conv3.shape)
            print("ac_input: ", self.ac_input.shape)
            print("fc1: ", self.fc1.shape)
            print("lstm_hidden_state_input: ", self.lstm_hidden_state_input.shape)
            print("lstm_input: ", self.lstm_input.shape)
            print("lstm_hidden_state_output: ", self.lstm_hidden_state_output.shape)
            print("lstm_output: ", self.lstm_output.shape)
            print("lstm_output_flat: ", self.lstm_output_flat.shape)
            print("value_output: ", self.value_output.shape)
            print("policy_logits: ", self.policy_logits.shape)
            print("policy_output: ", self.policy_output.shape)
            print("value_output_unflat: ", self.value_output_unflat.shape)
            print("policy_logits_unflat: ", self.policy_logits_unflat.shape)


            print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            graph = tf.get_default_graph()

            # TODO: get this some other way?
            conv1_w_summary = tf.summary.histogram('conv1 weights',graph.get_tensor_by_name("main_acn/Conv/weights:0"))
            conv1_b_summary = tf.summary.histogram('conv1 biases',graph.get_tensor_by_name("main_acn/Conv/biases:0"))
            conv2_w_summary = tf.summary.histogram('conv2 weights',graph.get_tensor_by_name("main_acn/Conv_1/weights:0"))
            conv2_b_summary = tf.summary.histogram('conv2 biases',graph.get_tensor_by_name("main_acn/Conv_1/biases:0"))
            conv3_w_summary = tf.summary.histogram('conv2 weights',graph.get_tensor_by_name("main_acn/Conv_2/weights:0"))
            conv3_b_summary = tf.summary.histogram('conv3 biases',graph.get_tensor_by_name("main_acn/Conv_2/biases:0"))
            fc1_w_summary = tf.summary.histogram('fc1 weights',graph.get_tensor_by_name("main_acn/fully_connected/weights:0"))
            fc1_b_summary = tf.summary.histogram('fc1 biases',graph.get_tensor_by_name("main_acn/fully_connected/biases:0"))
            lstm_w_summary = tf.summary.histogram('lstm weights',graph.get_tensor_by_name("main_acn/rnn/lstm_cell/kernel:0"))
            lstm_b_summary = tf.summary.histogram('lstm biases',graph.get_tensor_by_name("main_acn/rnn/lstm_cell/bias:0"))
            value_w_summary = tf.summary.histogram('value weights',graph.get_tensor_by_name("main_acn/fully_connected_1/weights:0"))
            value_b_summary = tf.summary.histogram('value biases',graph.get_tensor_by_name("main_acn/fully_connected_1/biases:0"))
            policy_w_summary = tf.summary.histogram('policy weights',graph.get_tensor_by_name("main_acn/fully_connected_2/weights:0"))
            policy_b_summary = tf.summary.histogram('policy biases',graph.get_tensor_by_name("main_acn/fully_connected_2/biases:0"))

            # Tensorboard
            self.average_reward_metric = tf.placeholder(tf.float32, name="average_reward")
            # self.average_length_of_episode = tf.placeholder(tf.float32, name="average_length_of_episode")

            conv1_summary = tf.summary.histogram('conv1', self.conv1)
            conv2_summary = tf.summary.histogram('conv2', self.conv2)
            conv3_summary = tf.summary.histogram('conv3', self.conv3)
            fc1_summary = tf.summary.histogram('fc1', self.fc1)
            # lstm_summary = tf.summary.histogram('lstm', self.lstm_cell)

            policy_summary = tf.summary.tensor_summary('policy', self.policy_output)
            reward_summary = tf.summary.scalar('average_reward_metric', self.average_reward_metric)
            loss_summary = tf.summary.scalar('loss', self.loss)
            entropy_summary = tf.summary.scalar('policy_entropy', tf.math.reduce_mean(self.extra.entropy))
            baseline_loss_summary = tf.summary.scalar('baseline_loss', tf.math.reduce_mean(self.extra.baseline_loss))
            entropy_loss_summary = tf.summary.scalar('entropy_loss', tf.math.reduce_mean(self.extra.entropy_loss))
            policy_gradient_loss = tf.summary.scalar('policy_gradient_loss', tf.math.reduce_mean(self.extra.policy_gradient_loss))
            
            self.train_step_summary = tf.summary.merge([
                reward_summary,
                loss_summary,
                entropy_summary,
                baseline_loss_summary,
                entropy_loss_summary,
                policy_gradient_loss,
                conv1_w_summary,
                conv1_b_summary,
                conv2_w_summary,
                conv2_b_summary,
                conv3_w_summary,
                conv3_b_summary,
                fc1_w_summary,
                fc1_b_summary,
                lstm_w_summary,
                lstm_b_summary,
                value_w_summary,
                value_b_summary,
                policy_w_summary,
                policy_b_summary
                ])

            self.action_step_summary = tf.summary.merge([policy_summary])
            # tf.summary.scalar('average_length_of_episode', self.average_length_of_episode)




    def get_action(self, sess, state, t_list, hidden_state_input, action_list, reward_list):
        """ 
        Feed forward to get action. 
        """

        # Add the extra dimension for feed forward
        if np.array(action_list).ndim == 1:
            action_list = np.expand_dims(action_list, axis=0)
        if np.array(reward_list).ndim == 1:
            reward_list = np.expand_dims(reward_list, axis=0)

        print("GET ACTION")
        print("action_list: ", action_list)
        print("reward_list: ", reward_list)

        feed = {self.inputs_: np.reshape(state, [-1, state_size[0], state_size[1], state_size[2]]),
                self.actions: action_list,
                self.rewards: reward_list}

        # Can also do placeholder with default
        if hidden_state_input is not None:
            feed[self.lstm_hidden_state_input] = hidden_state_input

        # policies, hidden_state_output = sess.run([mainA2C.policy_output, mainA2C.lstm_hidden_state_output], feed_dict=feed)
        actions, logits, policy, hidden_state_output, action_step_summary = \
                                                    sess.run([self.action_output, 
                                                            self.policy_logits, 
                                                            self.policy_output,
                                                            self.lstm_hidden_state_output,
                                                            self.action_step_summary], feed_dict=feed)
        print("logits: \n", logits)
        print("policy: \n", policy)
        return actions, hidden_state_output, action_step_summary
    
    def get_value(self, sess, state, hidden_state_input, action, reward):
        """ 
        Feed forward to get the value. 
        """
        print("GET_VALUE")
        print("state.shape: ", state.shape)
        feed = {self.inputs_: np.reshape(state, [-1, state_size[0], state_size[1], state_size[2]]),
                self.lstm_hidden_state_input: hidden_state_input,
                self.actions: np.expand_dims(action, axis=0),
                self.rewards: np.expand_dims(reward, axis=0)}
        values = sess.run(self.value_output, feed_dict=feed)
        values = np.array(values).flatten()
        return values

    def train_step(self, sess, states, actions, rewards, discounts, initial_Rs, hidden_state_input):
        """
        Backprop to get the loss.
        Done on partial trajectories.
        """
        print("TRAIN_STEP")
        loss, extra, opt, train_step_summary = \
                sess.run([self.loss, self.extra, self.opt, self.train_step_summary], 
                    feed_dict={self.inputs_: np.reshape(states, [-1, 80, 80, 3]),
                                self.actions: actions,
                                self.rewards: rewards,
                                self.discounts: discounts,
                                self.initial_Rs: initial_Rs,
                                self.lstm_hidden_state_input: hidden_state_input,
                                self.average_reward_metric: np.mean(rewards)})
        return loss, extra, train_step_summary

def _action(*entries):
    return np.array(entries, dtype=np.intc)

def get_random_action():
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
    env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED', 'POS'], config=config)
    env.reset()
    print("Started another environment worker!")

    while True:
        # if child_conn.poll():
        # Note: using the above loops, without it blocks. Not sure which is fastest.
            action, t = child_conn.recv()

            # Get position and velocityt
            if action == None:
                package = env.observations()['POS']
            else:
                package = env_step(env, action, t)
            child_conn.send(package)

def env_step(env, action, t, num_repeats=20):

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
            # reward = -1
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

def get_bootstrap(args, sess, mainA2C, hidden_state_input):
    args = np.moveaxis(args, -1, 0)
    # Getting R to use as initial condition. Essentially doing the whole target function thing. 
    _state, action, next_state, reward, _t, _episode_done = args
    next_state = np.array([np.array(s) for s in next_state])

    next_state_data = np.expand_dims(np.array(next_state), axis=0)
    bootstrapped_R = mainA2C.get_value(sess, next_state_data, hidden_state_input, action, reward)

    bootstrapped_Rs = []
    for i in range(num_envs):
        if reward[i] == 0:
            bootstrapped_Rs.append(bootstrapped_R[i])
        else:
            bootstrapped_Rs.append(reward[i])

    return bootstrapped_R

def deep_cast_to_nparray(bad_array):
    return np.array([np.array([np.array(a) for a in inner]) for inner in bad_array])

# def get_discounts(reward_list):
#   f = lambda x: 0.0 if x != 0 else gamma
#   return np.array([[f(x) for x in y] for y in reward_list])

def get_discounts(reward_list):
    return np.array([[gamma for x in y] for y in reward_list])

def get_positions(parent_conns):

    # Send empty package to indicate want position.
    parent_conns[0].send((None, None))
    position = parent_conns[0].recv()
    return position

def train(level, config, tensorboard_path, mainA2C):
    # Now train with experiences
    print("num_envs", num_envs)
    print("tensorboard_path", tensorboard_path)

    # Initialization

    # TODO: this should come from 
    envs_list = [deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config)] * num_envs
    envs_list = map(reset_envs, envs_list)
    state_batch = map(lambda env: env.observations()['RGB_INTERLEAVED'], envs_list)
    next_state_batch = copy.deepcopy(state_batch)

    # Init stuff for positions
    position_data = []

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
        # Need to switch for om
        # train_writer = tf.summary.FileWriter( '/mnt/hgfs/ryanprinster/lab/tensorboard', sess.graph)
        train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

        step = 0 # Same step at every train iteration
        t_list = [0 for i in range(num_envs)]
        action_list = np.random.randint(6, size=num_envs)
        reward_list = np.zeros(num_envs)
            
        for ep in range(1, train_episodes):

            # n-steps
            n_steps_parallel = []
            hidden_state_input = None
            for i in range(n):
                state_batch = next_state_batch # TODO: Namespace collision?

                # Track X, Y position
                position_data.append(get_positions(parent_conns))

                step += 1
                print("step: ", step)
                # GPU, PARALLEL
                action_list, hidden_state_input, action_step_summary = mainA2C.get_action(
                    sess, 
                    np.array(state_batch), 
                    t_list, 
                    hidden_state_input, 
                    action_list, 
                    reward_list)
                train_writer.add_summary(action_step_summary, step)

                print("action_list: ", action_list)

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


            # Need to this in GPU parallel
            bootstrap_vals_list = get_bootstrap(n_steps_parallel[0], sess, mainA2C, hidden_state_input)

            n_steps_parallel = [deep_cast_to_nparray(tup) for tup in np.moveaxis(n_steps_parallel, -1, 0)]
            state_list_train, action_list_train, _next_state_list, reward_list_train, _t_list, _episode_done_list = n_steps_parallel

            pcontinues_list = get_discounts(reward_list_train)

            print("action_list_train.shape: ", action_list_train)
            print("state_list_train.shape: ", state_list_train.shape)
            print("reward_list_train: ", reward_list_train)
            print("bootstrap_vals_list: ", bootstrap_vals_list)
            print("pcontinues_list: ", pcontinues_list)

            # Train step
            loss, extra, summary = mainA2C.train_step(sess, 
                state_list_train, 
                action_list_train, 
                reward_list_train, 
                pcontinues_list, 
                bootstrap_vals_list, 
                hidden_state_input)

            train_writer.add_summary(summary, int(step))
            
            print("total loss: ", loss)
            print("entropy: ", extra.entropy)
            print("entropy_loss: ", extra.entropy_loss)
            print("baseline_loss: ", extra.baseline_loss)
            print("policy_gradient_loss: ", extra.policy_gradient_loss)
            print("advantages: ", np.reshape(extra.advantages, [n, num_envs]))
            print("discounted_returns: ", np.reshape(extra.discounted_returns, [n, num_envs]))

            # TODO:
            # 1) Clip rewards
            # 2) Make min/max policy?
            # 3) clip policy gradients? Might already be done
            # 4) remove 0s in pcontinues?

            print("saving text")
            np.save('/mnt/hgfs/ryanprinster/test/position_data.npy', np.array(position_data))
        # print("Saving...")
        # saver.save(sess, '/mnt/hgfs/ryanprinster/lab/models/my_model', global_step=ep)

        # print("Resoring...")
        # saver.restore(sess, tf.train.latest_checkpoint('/mnt/hgfs/ryanprinster/lab/models/'))



def run(length, width, height, fps, level, record, demo, demofiles, video, 
    tensorboard_path, num_envs_, n_, max_steps_, learning_rate_, gamma_, 
      entropy_reg_term_):
  """Spins up an environment and runs the random agent."""
  # TODO: make tabs/spaces consistent
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

  # TODO: Remove global variables
  global max_steps, learning_rate, gamma, entropy_reg_term, num_envs, n
  max_steps, learning_rate, gamma, entropy_reg_term, num_envs, n = \
  max_steps_, learning_rate_, gamma_, entropy_reg_term_,  num_envs_, n_

  print('num_envs:', num_envs)
  print('n:', n)
  tf.reset_default_graph()
  mainA2C = ActorCriticNetwork(name='main_acn', num_envs=num_envs, n=n)

  train(level, config, tensorboard_path, mainA2C)


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
                      default='tests/hannahs_maze',
                      help='The environment level script to load')
  parser.add_argument('--record', type=str, default=None,
                      help='Record the run to a demo file')
  parser.add_argument('--demo', type=str, default=None,
                      help='Play back a recorded demo file')
  parser.add_argument('--demofiles', type=str, default=None,
                      help='Directory for demo files')
  parser.add_argument('--video', type=str, default=None,
                      help='Record the demo run as a video')
  parser.add_argument('--tensorboard_path', type=str, 
                      default='/mnt/hgfs/ryanprinster/lab/tensorboard',
                      help='Set the tensorboard path to save tensorboard output')
  parser.add_argument('--num_envs', type=int, default=1,
                      help='Set the number of environments to run in parallel')
  parser.add_argument('--n', type=int, default=5,
                      help='Set the length at which to truncate the LSTM')
  parser.add_argument('--max_steps', type=int, default=500,
                      help='Max number of steps before resetting the agent')
  parser.add_argument('--learning_rate', type=float, default=.001,
                      help='Learning rate')
  parser.add_argument('--gamma', type=float, default=.99,
                      help='Future reward discount')
  parser.add_argument('--entropy_reg_term', type=float, default=.05,
                      help='Entropy regularization term')


  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.length, args.width, args.height, args.fps, args.level_script,
      args.record, args.demo, args.demofiles, args.video, args.tensorboard_path, 
      args.num_envs, args.max_steps, args.n, args.learning_rate, args.gamma, 
      args.entropy_reg_term)
