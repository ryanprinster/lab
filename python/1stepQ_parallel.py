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
max_steps = 2000               # max steps before reseting the agent
gamma = 0.99                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.02            # exponential decay rate for exploration prob

# Network parameters
kernel_size_1 = [8,8,3]
output_filters_conv1 = 32     
output_filters_conv2 = 64     
output_filters_conv3 = 64     
hidden_size = 512               # number of units in each Q-network hidden layer
learning_rate = 0.000001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 4                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

minibatch_size = 5


#target QN
update_target_every = 40

state_size = (80,80,3)


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
            self.conv1 = tf.contrib.layers.conv2d(self.inputs_, output_filters_conv1, kernel_size=8, stride=2)
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
            self.discount = tf.constant(gamma,shape=[batch_size],dtype=tf.float32,name="discount")
      
            #TRFL qlearning
            qloss, q_learning = trfl.qlearning(self.output,self.actions_,self.reward,self.discount,self.targetQs_)
            self.loss = tf.reduce_mean(qloss)
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            
    def get_qnetwork_variables(self):
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]
    
    def get_action(self, sess, state):
        """ 
        Returns the action chosen by the QNetwork. 
        Should be called by the MainQN 
        """
        feed = {mainQN.inputs_: state}
        # feed = {mainQN.inputs_: state.reshape((batch_size, state.shape[0], state.shape[1], state.shape[2]))}
        Qs = sess.run(mainQN.output, feed_dict=feed)
        # print("Main Qs: ", Qs)
        action = np.argmax(Qs, axis=1)
        # action = np.ones(4)*4
        return action

    def get_targetQs(self, sess, next_states):
        """ 
        Returns the target Qs
        Should be called by targetQN 
        """
        return sess.run(self.output, feed_dict={self.inputs_: next_states})

    def train_step(self, sess, states, target_Qs, rewards, actions):
        """
        Runs a train step
        Returns the loss
        Should be called by MainQN
        """
        loss, _ = sess.run([self.loss, self.opt],
                    feed_dict={self.inputs_: states,
                               self.targetQs_: target_Qs,
                               self.reward: rewards,
                               self.actions_: actions})
        return loss

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
            package, reset = env_step(env, action, t)
            child_conn.send(package)
            if reset:
                env.reset()

def env_step(env, action, t, num_repeats=60):

    # print(index_to_english(action))
    english_action = index_to_english(action)
    action = map_to_dmlab(action)
    reward = 0
    count = 0
    reset = False
    while count < num_repeats:

      if not env.is_running():
        reset = True

      reward = env.step(action)
    
      if reward != 0:
        break #TODO
      
      count +=1
    if reward > 0:
        print("Action: ", english_action, " REWARD: " + str(reward), "Steps taken: ", t)

    # Dealing w/ end of episode
    next_state = None
    if reward > 0 or t == max_steps:
        next_state = np.zeros(state_size)
        t = 0
        reset = True
    else:
        next_state = env.observations()['RGB_INTERLEAVED']
        t += 1


    return (next_state, reward, t), reset

# def deal_with_end_of_episode(args):
#     state, reward, done, env, t, next_state = args
    
#     if done or t == max_steps-1:
#         # The episode ends so no next state
#         next_state = np.zeros(state.shape)
#         t = 0
#         # TODO: update global counter for number of episodes?

#         # Start new episode
#         env.reset()

#     else:
#         state = next_state
#         t += 1
#     return state, reward, done, env, t, next_state


def apply_epsilon_greedy(args):
    action, step = args
    
    # Explore or Exploit
    explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step)
    
    print("Explore P: ", explore_p)

    if explore_p > np.random.rand():
        # Replace action a random action
        action = get_random_action()
    # else, action is already gotten from network

    return action

def reset_envs(env):
    env.reset()
    return env


def train(level, config):
    # Now train with experiences

    # TODO: this is not used right now. Discount rewards
    # reward_list = [[]] * batch_size
    # Initialization
    envs_list = [deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config)] * batch_size
    envs_list = map(reset_envs, envs_list)
    state_list = map(lambda env: env.observations()['RGB_INTERLEAVED'], envs_list)
    minibatch_states_list = []
    minibatch_actions_list = []
    minibatch_targetQs_list = []
    minibatch_reward_list = []

    print(state_list[0].shape)

    # Initalization of multiprocessing stuff
    pipes = [Pipe() for i in range(batch_size)]
    parent_conns, child_conns = zip(*pipes)

    processes = [Process(target=env_worker, 
          args=(child_conns[i],level, config)) 
          for i in range(batch_size)]

    for i in range(batch_size):
        processes[i].start()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # https://medium.com/@anthony_sarkis/tensorboard-quick-start-in-5-minutes-e3ec69f673af
        # train_writer = tf.summary.FileWriter( '/mnt/hgfs/ryanprinster/lab/tensorboard', sess.graph)

        step = 0 # Same step at every train iteration
        total_rewards = [0] * batch_size
        t_list = [0] * batch_size

        # Profiling
        total_forwardprop = 0
        total_action = 0
        total_dealwendepisode = 0
        total_train = 0
            
        for ep in range(1, train_episodes):

            step += 1
            print("step: ", step)

            #update target q network
            if step % update_target_every == 0:
                sess.run(target_network_update_ops)
                print("\nCopied model parameters to target network.")


            # GPU, PARALLEL
            # Choose action according to an epsilon greedy policy
            # Batch size would change every time due to epsilon greedy.
            # Will parallelize next.

            start_forwardprop = time.time()

            action_list = mainQN.get_action(sess, np.array(state_list))
            action_list = map(apply_epsilon_greedy, zip(action_list, [step] * batch_size))

            end_fowardprop = time.time()
            total_forwardprop += end_fowardprop-start_forwardprop
            # print("mean forwardprop time:", total_forwardprop/step)


            # CPU, PARALLEL
            # Take action in environment, get new state and reward
            for i in range(batch_size):
                package = (action_list[i], t_list[i])
                parent_conns[i].send(package)

            nextstate_reward_t_list = [parent_conns[i].recv() for i in range(batch_size)]
            

            end_take_action = time.time()
            total_action += end_take_action - end_fowardprop
            # print("mean take action time: ", total_action/step)


            # Update rewards for all environments
            # [total_rewards[i] + nextstate_reward_done_list[i][1] for i in range(batch_size)]
            #  TODO: update total rewards as we go

            next_state_list, reward_list, t_list = zip(*nextstate_reward_t_list)

            # CPU, SERIAL (easily changed to parallel)
            # Inputs: state, reward, done, env, t, next_state
            # state_list, reward_list, done_list, env_list, t_list, next_state_list \
            #     = zip(*map(deal_with_end_of_episode, 
            #         zip(state_list,
            #             reward_list,
            #             done_list,
            #             envs_list,
            #             t_list,
            #             nextstate_list
            #         )))

            end_dealwith_endofepisode = time.time()
            total_dealwendepisode += end_dealwith_endofepisode - end_take_action
            # print("mean deal with end of episode time: ", total_dealwendepisode/step)

            # GPU, PARALLEL
            target_Qs = targetQN.get_targetQs(sess, np.array(next_state_list))
            # print("Target Qs: ", target_Qs)

            # Set target_Qs to 0 for states where episode ends
            # TODO: possible bug with the batch size?
            episode_ends = (next_state_list == np.zeros(state_list[0].shape)).all(axis=(1,2,3))
            target_Qs[episode_ends] = _action(0, 0, 0, 0, 0, 0)

            # GPU, PARALLEL
            # Train step
            if step % minibatch_size == 0 and step != 0:
                loss = mainQN.train_step(sess, minibatch_states_list, minibatch_targetQs_list, minibatch_reward_list, minibatch_actions_list)
                minibatch_states_list = []
                minibatch_actions_list = []
                minibatch_targetQs_list = []
                minibatch_reward_list = []
                # need to do for rewrards , actions, arget_qs
            else:
                if len(minibatch_states_list) == 0:
                    minibatch_states_list = state_list
                    minibatch_actions_list = action_list
                    minibatch_targetQs_list = target_Qs
                    minibatch_reward_list = reward_list
                else:
                    minibatch_states_list = np.concatenate((state_list, minibatch_states_list), axis=0)
                    minibatch_actions_list = np.concatenate((action_list, minibatch_actions_list), axis=0)
                    minibatch_targetQs_list = np.concatenate((target_Qs, minibatch_targetQs_list), axis=0)
                    minibatch_reward_list = np.concatenate((reward_list, minibatch_reward_list), axis=0)


            state_list = next_state_list

            # loss = mainQN.train_step(sess, state_list, target_Qs, reward_list, action_list)

            # state_list = next_state_list

            end_target_train = time.time()
            total_train += end_target_train - end_dealwith_endofepisode
            # print("mean train_time: ", total_train/step)
            
            # print("total forwardprop time:", total_forwardprop)
            # print("total take action time: ", total_action)
            # print("total deal with end of episode time: ", total_dealwendepisode)
            # print("total train_time: ", total_train)

            # if ep % 100 == 0:
            #     print("Ep: ", ep, ", Loss: ", loss)


        # print("Saving...")
        # saver.save(sess, '/mnt/hgfs/ryanprinster/lab/models/my_model', global_step=ep)

        # print("Resoring...")
        # saver.restore(sess, tf.train.latest_checkpoint('/mnt/hgfs/ryanprinster/lab/models/'))


tf.reset_default_graph()

mainQN = QNetwork(name='main_qn', hidden_size=hidden_size, learning_rate=learning_rate,batch_size=batch_size*(minibatch_size-1))
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
  # env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED', 'DEBUG.CAMERA.TOP_DOWN'], config=config)

  # env.reset()

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
