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
import time

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

train_episodes = 500           # max number of episodes to learn from max_steps = 5000               # max steps in an episode
gamma = 0.99                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.002            # exponential decay rate for exploration prob

# Network parameters
kernel_size_1 = [8,8,3]
output_filters_conv1 = 32     
output_filters_conv2 = 64     
output_filters_conv3 = 64     
hidden_size = 512               # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 1                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

#target QN
update_target_every = 2000



def _action(*entries):
  return np.array(entries, dtype=np.intc)

class QNetwork:
    def __init__(self, name, learning_rate=0.01, state_size=[80,80,3], 
                 action_size=6, hidden_size=10, batch_size=20):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size[0], state_size[1], state_size[2]], name='inputs')
            
            # Actions for the QNetwork:
            # One-hot vector, with each action being as follows:
            # (look_left, look_right, strafe_left, strafe_right, forward, backward)           
            # These are mapped to the deepmind-lab (not one-hot) actions with the same names
            # defined in ACTIONS


            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [batch_size], name='actions')
            # one_hot_actions = tf.one_hot(self.actions_, action_size)
            
            # Target Q values for training
            # self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            
            # ReLU hidden layers
            self.conv1 = tf.contrib.layers.conv2d(self.inputs_, output_filters_conv1, kernel_size=8, stride=4)
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, output_filters_conv2, kernel_size=4, stride=2)
            self.conv3 = tf.contrib.layers.conv2d(self.conv2, output_filters_conv3, kernel_size=4, stride=1)
            
            self.fc1 = tf.contrib.layers.fully_connected( \
              tf.reshape(self.conv3, [-1, self.conv3.shape[1]*self.conv3.shape[2]*self.conv3.shape[3]]), \
              hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc1, action_size, 
                                                            activation_fn=None)
            
            # tf.summary.histogram("output", self.output)

            print("Network shapes:")
            print(self.conv1.shape)
            print(self.conv2.shape)
            print(self.conv3.shape)
            print(self.fc1.shape)
            print(self.output.shape)

            self.name = name

            #TRFL way
            self.targetQs_ = tf.placeholder(tf.float32, [batch_size,action_size], name='target')
            self.reward = tf.placeholder(tf.float32,[batch_size],name="reward")
            self.discount = tf.constant(0.99,shape=[batch_size],dtype=tf.float32,name="discount")
      
            # print(self.output.shape)
            # print(self.actions_.shape)
            # print(self.reward.shape)
            # print(self.discount.shape)
            # print(self.targetQs_.shape)
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



# Profiling Variables
# TODO: Make more clever profiling scheme
total_steps = 0
total_step_time = 0
mean_step_time = 0

total_network_updates = 0
total_network_update_time = 0
mean_network_update_time = 0

start_program = 0

def env_step(env, action, num_repeats=60):
    print(index_to_english(action))
    action = map_to_dmlab(action)
    reward = 0
    count = 0
    while count < num_repeats:

      if not env.is_running():
        env.reset()

      # Profile environment step
      global total_steps, total_step_time, mean_step_time
      start = time.clock()
      reward = env.step(action)
      step_time = time.clock() - start
      total_step_time += step_time
      total_steps += 1
      mean_step_time = total_step_time/total_steps

      if reward != 0:
        print("REWARD: " + str(reward))
        break #TODO

      count +=1

    done = reward > 0
    next_state = env.observations()['RGB_INTERLEAVED']
    # next_state = np.reshape(next_state, [-1])

    return next_state, reward, done


def pretrain(env, memory):
    state, reward, done = env_step(env, get_random_action())

    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length):


        action = get_random_action()
        next_state, reward, done = env_step(env, action)
 
        if done:
            # The simulation fails so no next state
            next_state = np.zeros(state.shape)
            # Add experience to memory
            memory.add((state, action, reward, next_state))

            # Start new episode
            env.reset()
            # Take one random step to get the pole and cart moving
            # state, reward, done = env_step(env, get_random_action())


        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            state = next_state
    
    return state


# class Profiler:
#     def __init__(self):
#         self.d = {}

#     def time(name, mode="interval"):
#         """
#         Function used to profile code. When called the first time
#         """
#         if name not in self.d:
#             self.d = time.clock()
#             # return

#         elif mode == "interval":
#             interval = time.clock() - self.d[]
#             self.d = 





def train(env, memory, state):
    # Now train with experiences
    global start_program
    start_program = time.clock()
    saver = tf.train.Saver()
    rewards_list = []
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # https://medium.com/@anthony_sarkis/tensorboard-quick-start-in-5-minutes-e3ec69f673af
        # train_writer = tf.summary.FileWriter( '/mnt/hgfs/ryanprinster/lab/tensorboard', sess.graph)

        step = 0
        for ep in range(1, train_episodes):

            total_program_time = time.clock() - start_program
            print("Mean step time: ", mean_step_time)
            print("Mean network update time: ", mean_network_update_time)
            print("Total step time: ", total_step_time)
            print("Total network update time: ", total_network_update_time)
            print("The rest of the program time: ", total_program_time-(total_step_time + total_network_update_time))

            total_reward = 0
            t = 0
            while t < max_steps:
                step += 1

                # End episode when hitting max_steps
                if t == max_steps-1:
                  done = True

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
                    #  Add profiling
                    global total_network_update_time, total_network_updates, mean_network_update_time
                    start = time.clock()

                    # Get action from Q-network
                    feed = {mainQN.inputs_: state.reshape((1, state.shape[0], state.shape[1], state.shape[2]))}
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)

                    network_update_time = time.clock() - start
                    total_network_update_time += network_update_time
                    total_network_updates += 1
                    mean_network_update_time = total_network_update_time/total_network_updates


                # Take action, get new state and reward

                next_state, reward, done = env_step(env, action)

                total_reward += reward

                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
                    t = max_steps
                    
                    print('Episode: {}'.format(ep),
                          'Total reward: {}'.format(total_reward),
                          # 'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_p))
                    rewards_list.append((ep, total_reward))

                    # Add experience to memory

                    memory.add((state, action, reward, next_state))

                    # Start new episode
                    env.reset()
                    # Take one random step to get the pole and cart moving
                    # state, reward, done = env_step(env, get_random_action())

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



                # Train and profile network
                global total_network_update_time, total_network_updates, mean_network_update_time
                start = time.clock()
                target_Qs = sess.run(targetQN.output, feed_dict={targetQN.inputs_: next_states})
                network_update_time = time.clock() - start
                total_network_update_time += network_update_time
                total_network_updates += 1
                mean_network_update_time = total_network_update_time/total_network_updates


                # Set target_Qs to 0 for states where episode ends
                # TODO: This is kinda weird with the mapping. 
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=(1,2,3))
                target_Qs[episode_ends] = _action(0, 0, 0, 0, 0, 0)

                #TRFL way, calculate td_error within TRFL
                # Profiling
                # merge = tf.summary.merge_all()

                global total_network_update_time, total_network_updates, mean_network_update_time
                start = time.clock()

                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                    feed_dict={mainQN.inputs_: states,
                                               mainQN.targetQs_: target_Qs,
                                               mainQN.reward: rewards,
                                               mainQN.actions_: actions})

                network_update_time = time.clock() - start
                total_network_update_time += network_update_time
                total_network_updates += 1
                mean_network_update_time = total_network_update_time/total_network_updates

                # train_writer.add_summary(summary, t)

            # print("Saving...")
            # saver.save(sess, '/mnt/hgfs/ryanprinster/lab/models/my_model', global_step=ep)

            # print("Resoring...")
            # saver.restore(sess, tf.train.latest_checkpoint('/mnt/hgfs/ryanprinster/lab/models/'))


tf.reset_default_graph()
mainQN = QNetwork(name='main_qn', hidden_size=hidden_size, learning_rate=learning_rate,batch_size=batch_size)
targetQN = QNetwork(name='target_qn', hidden_size=hidden_size, learning_rate=learning_rate,batch_size=batch_size)

target_network_update_ops = trfl.update_target_variables(targetQN.get_qnetwork_variables(),mainQN.get_qnetwork_variables(),tau=1.0)







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

  # ACTION_LIST = list(six.viewvalues(ACTIONS))
  
  # random_action = random.choice(ACTION_LIST)
  
  # print(env.action_spec())

  # state, reward, done = env_step(env, get_random_action())
  # print(reward)

  # obs = env.observations()
  # print(obs)
    
  # ACTUALLY RUNNING STUFF

  # print("INPUTS SHAPE:")
  # print(tf.shape(mainQN.inputs_))

  # Initialize the simulation
  env.reset()
  # Take one random step to get the pole and cart moving

  memory = Memory(max_size=memory_size)



  state = pretrain(env, memory)
  train(env, memory, state)




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
