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
"""Basic random agent for DeepMind Lab."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import six

import deepmind_lab

""" Dependencies used for this:
    pip uninstall numpy
    pip install --no-cache-dir numpy==1.15.4
    pip install --upgrade tensorflow
    pip install --upgrade tensorflow-probability
    pip install wrapt"""
import tensorflow as tf
import trfl

def _action(*entries):
  return np.array(entries, dtype=np.intc)

class QNetwork:
    def __init__(self, name, learning_rate=0.01, state_size=4, 
                 action_size=2, hidden_size=10, batch_size=20):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [batch_size], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)
            
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            
            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size, 
                                                            activation_fn=None)
            
            self.name = name

            #TRFL way
            self.targetQs_ = tf.placeholder(tf.float32, [batch_size,action_size], name='target')
            self.reward = tf.placeholder(tf.float32,[batch_size],name="reward")
            self.discount = tf.constant(0.99,shape=[batch_size],dtype=tf.float32,name="discount")
      
            #TRFL qlearning
            qloss, q_learning = trfl.qlearning(self.output,self.actions_,self.reward,self.discount,self.targetQs_)
            self.loss = tf.reduce_mean(qloss)
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            
    def get_qnetwork_variables(self):
      return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]
    """
    def get_action(self, sess):
      #Returns the action chosen by the QNetwork. Should be called by the MainQN
      feed = {self.inputs_: state.reshape((1, *state.shape))}
      Qs = sess.run(self.output, feed_dict=feed)
      action = np.argmax(Qs)
      return action
    """

from collections import deque
class Memory():
    def __init__(self, max_size = 1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]


def get_random_action():

    # Make a random action
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

    ACTION_LIST = list(six.viewvalues(ACTIONS))

    return random.choice(ACTION_LIST)


def pretrain():
    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length):
        # Uncomment the line below to watch the simulation
        # env.render()

        # TODO: step only returns reward

        reward = env.step(get_random_action())
        done = not env.is_running()
        next_state = env.observations()['RGB_INTERLEAVED']


        if done:
            # The simulation fails so no next state
            next_state = np.zeros(state.shape)
            # Add experience to memory
            memory.add((state, action, reward, next_state))

            # Start new episode
            env.reset()
            # Take one random step to get the pole and cart moving
            reward = env.step(get_random_action())
            done = not env.is_running()
            next_state = env.observations()['RGB_INTERLEAVED']

        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            state = next_state

def train():
    # Now train with experiences
    #saver = tf.train.Saver()
    rewards_list = []
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        step = 0
        for ep in range(1, train_episodes):
            total_reward = 0
            t = 0
            while t < max_steps:
                step += 1
                # Uncomment this next line to watch the training
                # env.render()

                #update target q network
                if step % update_target_every == 0:
                    #TRFL way
                    sess.run(target_network_update_ops)
                    print("\nCopied model parameters to target network.")

                # Explore or Exploit
                explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step)
                if explore_p > np.random.rand():
                    # Make a random action
                    action = get_random_action()
                else:
                    # Get action from Q-network
                    feed = {mainQN.inputs_: state.reshape((1, state.shape[0]))}
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)

                # Take action, get new state and reward

                reward = env.step(action)
                done = not env.is_running()
                next_state = env.observations()['RGB_INTERLEAVED']

                total_reward += reward

                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
                    t = max_steps

                    print('Episode: {}'.format(ep),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_p))
                    rewards_list.append((ep, total_reward))

                    # Add experience to memory
                    memory.add((state, action, reward, next_state))

                    # Start new episode
                    env.reset()
                    # Take one random step to get the pole and cart moving

                    reward = env.step(get_random_action())
                    done = not env.is_running()
                    next_state = env.observations()['RGB_INTERLEAVED']

                else:
                    # Add experience to memory
                    memory.add((state, action, reward, next_state))
                    state = next_state
                    t += 1

                # Sample mini-batch from memory
                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])

                # Train network
                #in this example (and in Deep Q Networks) use targetQN for the target values
                #target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})
                target_Qs = sess.run(targetQN.output, feed_dict={targetQN.inputs_: next_states})

                # Set target_Qs to 0 for states where episode ends
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                target_Qs[episode_ends] = (0, 0)

                #TRFL way, calculate td_error within TRFL
                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                    feed_dict={mainQN.inputs_: states,
                                               mainQN.targetQs_: target_Qs,
                                               mainQN.reward: rewards,
                                               mainQN.actions_: actions})



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
  env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED', 'DEBUG.CAMERA.TOP_DOWN'], config=config)

  env.reset()

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

  ACTION_LIST = list(six.viewvalues(ACTIONS))
  
  random_action = random.choice(ACTION_LIST)
  
  print(env.action_spec())

  reward = env.step(random_action)
  print(reward)

  obs = env.observations()
  print(obs)
    
  pretrain()
  train()

  train_episodes = 500         # max number of episodes to learn from
  max_steps = 200                # max steps in an episode
  gamma = 0.99                   # future reward discount

  # Exploration parameters
  explore_start = 1.0            # exploration probability at start
  explore_stop = 0.01            # minimum exploration probability 
  decay_rate = 0.0002            # exponential decay rate for exploration prob
  
  # Network parameters
  hidden_size = 64               # number of units in each Q-network hidden layer
  learning_rate = 0.0001         # Q-network learning rate
  
  # Memory parameters
  memory_size = 10000            # memory capacity
  batch_size = 20                # experience mini-batch size
  pretrain_length = batch_size   # number experiences to pretrain the memory
  
  #target QN
  update_target_every = 2000
  
  tf.reset_default_graph()
  mainQN = QNetwork(name='main_qn', hidden_size=hidden_size, learning_rate=learning_rate,batch_size=batch_size)
  targetQN = QNetwork(name='target_qn', hidden_size=hidden_size, learning_rate=learning_rate,batch_size=batch_size)
  
  target_network_update_ops = trfl.update_target_variables(targetQN.get_qnetwork_variables(),mainQN.get_qnetwork_variables(),tau=1.0)
  
  # Initialize the simulation
  env.reset()
  # Take one random step to get the pole and cart moving
  state, reward, done, _ = env.step(env.action_space.sample())
  
  print(type(state))

  memory = Memory(max_size=memory_size)




if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--length', type=int, default=1000,
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
