"""
Trainers are the actual training loops of different modules. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import six
import itertools
import sys
import tensorflow as tf

from multiprocessing import Process, Queue, Pipe

import deepmind_lab
from environment import ParallelEnv
from cells import PlaceCells, HeadDirCells
from grid_network import GridNetwork, VisionModule, GridNetworkAgent
from rat_trajectory_generator import RatTrajectoryGenerator as Rat
from replay_buffer import ReplayBuffer, BigReplayBuffer
from a2c_lstm_2 import A2CAgent 

class Trainer(object):
    """
    Class meant to run the training pipeline for the grid network with one set
    of hyper parameters.
    """
    def __init__(self, 
        base_path,
        unique_exp_name, # string indicating experiment
        restore=False,
        N=256, 
        M=12, 
        trajectory_length=100, 
        train_iterations=5000, 
        learning_rate=1e-3,
        batch_size=16, 
        num_envs=4,
        sigma=1,
        n=5,
        gamma=.99,
        level_script='tests/empty_room_test', 
        obs_types=['RGB_INTERLEAVED', 'VEL.TRANS', 'VEL.ROT', 'POS',
                'DISTANCE_TO_WALL', 'ANGLE_TO_WALL', 'ANGLES'],
        config={'width': str(80), 'height': str(80)}):

        self.env = ParallelEnv(level_script, obs_types, config, num_envs)
        self.place_cells = PlaceCells(N, sigma)
        self.head_cells = HeadDirCells(M)
        self.rat = Rat(self.env, trajectory_length)
        self.grid_network = GridNetwork(name="grid_network", 
            learning_rate=learning_rate, max_time=trajectory_length)

        self.train_iterations = train_iterations
        self.batch_size = batch_size

        self.base_path = base_path
        self.experiments_save_path = self.base_path + 'experiments/'
        self.unique_exp_save_path = self.experiments_save_path \
            + 'exp_' + unique_exp_name + '/'

        self.restore = restore
        self.saver = tf.train.Saver()

    def train(self):
        print("Trainer.train")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Tensorflow saver stuff
            if self.restore:
                self.saver.restore(sess, \
                    tf.train.latest_checkpoint(self.unique_exp_save_path))

            # Tensorboard stuff
            train_writer = tf.summary.FileWriter(self.unique_exp_save_path)


            # Start A2C agent with a 32 process ParallelEnv
            # Start VisionModule with 1 process
            # Start GridNetwork with 1 process
            # Make general wrapper to start something else in a new process?
            # Start main Trainer, ReplayBuffer on main process, with shitton of memory

            # Start 

            for i in range(self.train_iterations):
                print("Train iter: ", i)
                sys.stdout.flush()

                data, actions, seeds = self.rat.generateAboutNTrajectories(self.batch_size)
                loss, summary = self.grid_network.train_step(sess, data, self.place_cells, 
                    self.head_cells)
                train_writer.add_summary(summary, i)

                print("Loss:", loss)

                if i % 10 == 0:
                    self.saver.save(sess, self.unique_exp_save_path)

                    histograms = self.grid_network.get_grid_layer_activations(sess, 
                        data, self.place_cells, self.head_cells)

                    print("And we have histograms:")
                    print(histograms.shape)

                    np.save(self.unique_exp_save_path + 'place_cell_histograms.npy', \
                        histograms)

            data, actions, seeds = self.rat.generateAboutNTrajectories(10*self.batch_size)
            histograms = self.grid_network.get_grid_layer_activations(sess, 
                data, self.place_cells, self.head_cells, pred_place_cell, pred_head_dir)



# class GridCellAgentTrainer(object):
#     """
#     Class meant to run the training pipeline for the full grid agent with one 
#     set of hyper parameters.

#     NOTE:
#     This version trains on generated trajectories instead of different 
#     """
#     def __init__(self, 
#         base_path,
#         unique_exp_name, # string indicating experiment
#         restore=False,
#         N=256, 
#         M=12, 
#         trajectory_length=100, 
#         train_iterations=5000, 
#         learning_rate=1e-3,
#         batch_size=16, 
#         num_envs=4,
#         sigma=1,
#         n=5,
#         gamma=.99,
#         level_script='tests/empty_room_test', 
#         obs_types=['RGB_INTERLEAVED', 'VEL.TRANS', 'VEL.ROT', 'POS',
#                 'DISTANCE_TO_WALL', 'ANGLE_TO_WALL', 'ANGLES'],
#         config={'width': str(80), 'height': str(80)}):

#         self.env_policy = ParallelEnv(level_script, obs_types, config, num_envs, num_steps=4)
#         self.place_cells = PlaceCells(N, sigma)
#         self.head_cells = HeadDirCells(M)
#         self.grid_network = GridNetworkAgent(batch_size=batch_size, name="grid_network", 
#             learning_rate=learning_rate, max_time=trajectory_length)
#         self.vision_module = VisionModule(name="vision_module", 
#             learning_rate=learning_rate, max_time=trajectory_length)

#         self.rat = Rat(self.env_policy, trajectory_length)
#         # self.a2c_agent = A2CAgent(level_script, num_envs, n, tensorboard_path, train_iterations, 
#         # gamma)
#         # self.replay_buffer = BigReplayBuffer(size=100)

#         self.train_iterations = train_iterations
#         self.batch_size = batch_size


#         self.base_path = base_path
#         self.experiments_save_path = self.base_path + 'experiments/'
#         self.unique_exp_save_path = self.experiments_save_path \
#             + 'exp_' + unique_exp_name + '/'

#         self.restore = restore
#         self.saver = tf.train.Saver()


#     def train(self):
#         # Start A2C agent with a 32 process ParallelEnv
#             # Extend A2C agent to add experience to replay
#         # Start VisionModule with 1 process
#         # Start GridNetwork with 1 process
#         # Make general wrapper to start something else in a new process?
#         # Start main Trainer, ReplayBuffer on main process, with shitton of memory

#         print("Trainer.train")

#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())

#             # Tensorflow saver stuff
#             if self.restore:
#                 self.saver.restore(sess, \
#                     tf.train.latest_checkpoint(self.unique_exp_save_path))

#             # Tensorboard stuff
#             train_writer = tf.summary.FileWriter(self.unique_exp_save_path)


#             for i in range(self.train_iterations):
#                 print("Train iter: ", i)
#                 sys.stdout.flush()
                
#                 data, actions, seeds = self.rat.generateAboutNTrajectories(self.batch_size)

#                 # self.replay_buffer.add(actions, seeds)
#                 # print(self.replay_buffer.trajectory_length)

#                 # Vision Network
#                 vision_loss, pred_place_cell, pred_head_dir, vision_summary = \
#                     self.vision_module.train_step(sess, data, self.place_cells, 
#                     self.head_cells)

#                 # Grid Network
#                 grid_loss, grid_summary = self.grid_network.train_step(sess, data, 
#                     self.place_cells, self.head_cells, pred_place_cell, pred_head_dir)

#                 train_writer.add_summary(grid_summary, i)
#                 train_writer.add_summary(vision_summary, i)
#                 print("grid_loss loss: ", np.sum(grid_loss))
#                 print("vision_loss loss: ", np.sum(vision_loss))
#                 sys.stdout.flush()

#                 # Save stuff every once in a while
#                 if i % 10 == 0:
#                     self.saver.save(sess, self.unique_exp_save_path)

#                     histograms = self.grid_network.get_grid_layer_activations(sess, 
#                         data, self.place_cells, self.head_cells, pred_place_cell, pred_head_dir)

#                     print("And we have histograms:")
#                     print(histograms.shape)

#                     np.save(self.unique_exp_save_path + 'place_cell_histograms.npy', \
#                         histograms)

#             # Final 
#             data, actions, seeds = self.rat.generateAboutNTrajectories(10*self.batch_size)
#             histograms = self.grid_network.get_grid_layer_activations(sess, 
#                 data, self.place_cells, self.head_cells, pred_place_cell, pred_head_dir)



if __name__ == '__main__':
    pass