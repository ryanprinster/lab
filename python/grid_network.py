# Copyright 2016-17 Google Inc.
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
################################################################################
"""Grid Cell Agent"""
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
# from python import rat_trajectory_generator, cells

"""
Util functions
"""
def transform_input_data(data, place_cells, head_cells):
    observation_data, position_data, direction_data, trans_velocity_data, \
        strafe_trans_velocity_data, ang_velocity_data = data 
    
    cos_vel_data = np.expand_dims(np.cos(ang_velocity_data), axis=2)
    sin_vel_data = np.expand_dims(np.sin(ang_velocity_data), axis=2)
    x_y_positions = np.delete(position_data,2,axis=2)
    trans_vel_data = np.expand_dims(trans_velocity_data, axis=2)
    strafe_trans_vel_data = np.expand_dims(strafe_trans_velocity_data, axis=2)
    direction_data = np.array(direction_data)

    place_cell_activity_labels = \
        place_cells.get_batched_ground_truth_activations(x_y_positions)
    head_cell_activity_labels = \
        head_cells.get_batched_ground_truth_activations(direction_data)

    initial_pos_activity = place_cell_activity_labels[:, 0, :]
    initial_dir_activity = head_cell_activity_labels[:, 0, :]

    # # Input data
    # print("cos_vel_data: ", cos_vel_data.shape)
    # print("sin_vel_data: ", sin_vel_data.shape)
    # print("trans_vel_data: ", trans_vel_data.shape)

    # # Labels / activities
    # print("x_y_positions: ", x_y_positions.shape)
    # print("direction_data: ", direction_data.shape)        
    # print("place_cell_activity_labels: ", place_cell_activity_labels.shape)
    # print("head_cell_activity_labels: ", head_cell_activity_labels.shape)        

    # # LSTM initiation
    # print("initial_pos_activity: ", initial_pos_activity.shape)
    # print("initial_dir_activity: ", initial_dir_activity.shape)

    return (trans_vel_data, strafe_trans_vel_data, cos_vel_data, sin_vel_data, initial_pos_activity, 
    initial_dir_activity, place_cell_activity_labels, 
    head_cell_activity_labels, x_y_positions)


class GridNetwork(object):
    """Grid Network"""
    def __init__(self, name, lstm_size=128, grid_layer_size=512, 
        N=256, M=12, learning_rate=1e-3, grad_clip_thresh=1e-5, max_time=100):

        # TODO: share these
        self.grid_layer_size = grid_layer_size

        
        with tf.variable_scope(name):
            
            # Constructing inputs to the lstm layer
            self.translational_velocity = tf.placeholder(tf.float32, [None, max_time, 1],
                name='trans_vel_input')
            self.sine_angular_velocity = tf.placeholder(tf.float32, [None, max_time, 1],
                name='sin_ang_vel_input')
            self.cosine_angular_velocity = tf.placeholder(tf.float32, [None, max_time, 1],
                name='cos_ang_vel_input')

            self.lstm_input = tf.concat([
                self.translational_velocity,
                self.sine_angular_velocity,
                self.cosine_angular_velocity,],
                axis=2, name='lstm_input')

            self.place_activity = tf.placeholder(tf.float32, [None, N], 
                name='place_activity_init')
            self.head_dir_activity = tf.placeholder(tf.float32, [None, M],
                name='head_activity_init')

            # Initial cell state and hidden state are linear transformations of 
            # place and head direction cells. Cell state and hidden state are 
            # concatenated for LSTMCell(state_is_tuple=False).

            # TODO: lstm_size/2 is arbitraty. should probably be proportion to N and M.
            # Actually should be same size as y and z?
            self.lstm_init_cell_hidden_state = tf.concat([
                    tf.contrib.layers.fully_connected(self.place_activity, int(lstm_size/2), activation_fn=None),
                    tf.contrib.layers.fully_connected(self.head_dir_activity, int(lstm_size/2), activation_fn=None),
                    tf.contrib.layers.fully_connected(self.place_activity, int(lstm_size/2), activation_fn=None),
                    tf.contrib.layers.fully_connected(self.head_dir_activity, int(lstm_size/2), activation_fn=None),
                ], axis=1, name='lstm_init_cell_hidden_state')

            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=False)

            print("self.lstm_init_cell_hidden_state.shape", self.lstm_init_cell_hidden_state.shape)

            # TODO: unroll lstm for 100 time steps (paper says so)
            self.lstm_output, self.lstm_hidden_state_output = tf.nn.dynamic_rnn(
                self.lstm_cell,
                self.lstm_input,
                initial_state=self.lstm_init_cell_hidden_state,
                dtype=tf.float32)
            
            # Grid Cell layer
            self.grid_cell_layer = tf.contrib.layers.fully_connected(\
                self.lstm_output, grid_layer_size, activation_fn=None)

            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
            self.dropout = tf.contrib.layers.dropout(self.grid_cell_layer, 
                keep_prob=self.dropout_prob)

            # Place and head cells
            self.pred_place_cell_logits = tf.contrib.layers.fully_connected(\
                self.dropout, N)

            self.pred_head_dir_logits = tf.contrib.layers.fully_connected(\
                self.dropout, M)

            # self.pred_place_cell = tf.contrib.layers.softmax(self.pred_place_cell_logits)
            # self.pred_head_dir = tf.contrib.layers.softmax(self.pred_head_dir_logits)

            # Construct loss
            self.place_cell_labels = tf.placeholder(tf.float32, [None, max_time, N],
                name='place_cell_labels')
            self.head_dir_labels = tf.placeholder(tf.float32, [None, max_time, M],
                name='head_dir_labels')

            self.place_cell_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.place_cell_labels,
                logits=self.pred_place_cell_logits)
            self.head_dir_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.head_dir_labels,
                logits=self.pred_head_dir_logits)

            self.loss = tf.reduce_mean(\
                tf.math.add(self.place_cell_loss, self.head_dir_loss))

            self.optimizer = tf.train.RMSPropOptimizer(\
                learning_rate=learning_rate, momentum=.9).minimize(self.loss)

            # # Optimizer
            # # TODO: Add weight decay to decoder layers
            # self.optimizer = tf.train.RMSPropOptimizer(\
            #     learning_rate=learning_rate, momentum=.9)
            
            # # Clip gradients
            # gvs = self.optimizer.compute_gradients(self.loss)
            # # Note: currently, last 6 ones here correspond to the grid cell layer
            # # and the place/head cell layers. 
            # capped_gvs = [(tf.clip_by_value(grad, -grad_clip_thresh, \
            #     grad_clip_thresh), var) \
            #     for grad, var in gvs[-6:]]

            # new_gvs = gvs[:-6] + capped_gvs

            # self.train_op = self.optimizer.apply_gradients(new_gvs)

            # Summary Stuff
            self.loss_summary = tf.summary.scalar('loss', self.loss)
            self.summary = tf.summary.merge_all()


    def train_step(self, sess, data, place_cells, head_cells):

        trans_vel_data, strafe_trans_velocity_data, cos_vel_data, sin_vel_data, initial_pos_activity, \
        initial_dir_activity, place_cell_activity_labels, \
        head_cell_activity_labels, x_y_positions = \
            transform_input_data(data, place_cells, head_cells)

        feed = {
            self.translational_velocity: trans_vel_data,
            self.cosine_angular_velocity: cos_vel_data,
            self.sine_angular_velocity: sin_vel_data,
            self.place_activity: initial_pos_activity,
            self.head_dir_activity: initial_dir_activity,
            self.dropout_prob: .5,
            self.place_cell_labels: place_cell_activity_labels,
            self.head_dir_labels: head_cell_activity_labels,
        }

        _, loss, summary = sess.run(
            [self.optimizer, self.loss, self.summary], 
            feed_dict=feed)

        return loss, summary

    def get_grid_layer_activations(self, sess, data, place_cells, head_cells):

        trans_vel_data, strafe_trans_velocity_data, cos_vel_data, sin_vel_data, initial_pos_activity, \
        initial_dir_activity, place_cell_activity_labels, \
        head_cell_activity_labels, x_y_positions = \
            transform_input_data(data, place_cells, head_cells)

        feed = {
            self.translational_velocity: trans_vel_data,
            self.cosine_angular_velocity: cos_vel_data,
            self.sine_angular_velocity: sin_vel_data,
            self.place_activity: initial_pos_activity,
            self.head_dir_activity: initial_dir_activity,
            self.dropout_prob: 0,
        }

        grid_layer_activations = sess.run(self.grid_cell_layer, feed_dict=feed)
        hist = self._grid_activations_to_histogram(grid_layer_activations, 
            x_y_positions)

        return hist

    def _grid_activations_to_histogram(self, grid_layer_activations, 
            x_y_positions, bins=32):

        grid_acts = np.reshape(grid_layer_activations, (-1, self.grid_layer_size))
        xy_pos = np.reshape(x_y_positions, (-1, 2))
        data = zip(xy_pos, grid_acts)        

        def xy_coord_to_ind(loc):
            # Assuming a valid location
            x_coord, y_coord = loc
            low, high = 100, 800
            x, y = x_coord-low, y_coord-low
            size_of_bin = (high-low)/bins
            x_ind, y_ind = int(x/(size_of_bin+1)), int(y/(size_of_bin+1)) #wont have anything in last bin really?
            return x_ind, y_ind

        def construct_histogram(data):
            #janky avoid /0
            counts = np.ones((self.grid_layer_size, bins, bins))*1e-10
            activations = np.zeros((self.grid_layer_size, bins, bins))
            for loc, acts in data:
                x_ind, y_ind = xy_coord_to_ind(loc)
                for i, act in enumerate(acts):
                    activations[i, x_ind, y_ind] += act
                    counts[i, x_ind, y_ind] += 1
            histograms = activations/counts
            return histograms

        return construct_histogram(data)

    # TODO: similar function for head direction cells 


class GridNetworkAgent(object):
    """Grid Network, modified slightly for the reinforcement learning setup"""
    def __init__(self, batch_size, name, lstm_size=128, grid_layer_size=512, 
        N=256, M=12, learning_rate=1e-3, grad_clip_thresh=1e-5, max_time=100):

        # TODO: share these
        self.name = name
        self.grid_layer_size = grid_layer_size
        self.batch_size = batch_size
        self.max_time = max_time

        # self.graph = tf.Graph()
        # with self.graph.as_default():
        with tf.variable_scope(name):

            # Constructing inputs to the lstm layer
            self.fwd_trans_velocity = tf.placeholder(tf.float32, [None, max_time, 1],
                name='fwd_trans_vel_input')
            self.strafe_trans_velocity = tf.placeholder(tf.float32, [None, max_time, 1],
                name='strafe_trans_vel_input')
            self.sine_angular_velocity = tf.placeholder(tf.float32, [None, max_time, 1],
                name='sin_ang_vel_input')
            self.cosine_angular_velocity = tf.placeholder(tf.float32, [None, max_time, 1],
                name='cos_ang_vel_input')

            # Inputs from the vision module
            self.y = tf.placeholder(tf.float32, [None, max_time, N])
            self.z = tf.placeholder(tf.float32, [None, max_time, M])

            self.lstm_input = tf.concat([
                self.fwd_trans_velocity,
                self.strafe_trans_velocity,
                self.sine_angular_velocity,
                self.cosine_angular_velocity,
                self.y,
                self.z,
                ],
                axis=2, name='lstm_input')

            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=False)
            self.lstm_init_cell_hidden_state = self.lstm_cell.zero_state(\
                batch_size=self.batch_size, dtype=tf.float32)

            self.lstm_output, self.lstm_hidden_state_output = tf.nn.dynamic_rnn(
                self.lstm_cell,
                self.lstm_input,
                initial_state=self.lstm_init_cell_hidden_state,
                dtype=tf.float32)
            
            # Grid Cell layer
            self.grid_cell_layer = tf.contrib.layers.fully_connected(\
                self.lstm_output, grid_layer_size, activation_fn=None)

            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
            self.dropout = tf.contrib.layers.dropout(self.grid_cell_layer, 
                keep_prob=self.dropout_prob)

            # Place and head cells
            self.pred_place_cell_logits = tf.contrib.layers.fully_connected(\
                self.dropout, N)

            self.pred_head_dir_logits = tf.contrib.layers.fully_connected(\
                self.dropout, M)

            # Construct loss
            self.place_cell_labels = tf.placeholder(tf.float32, [None, max_time, N],
                name='place_cell_labels')
            self.head_dir_labels = tf.placeholder(tf.float32, [None, max_time, M],
                name='head_dir_labels')

            self.place_cell_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.place_cell_labels,
                logits=self.pred_place_cell_logits)
            self.head_dir_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.head_dir_labels,
                logits=self.pred_head_dir_logits)

            self.loss = tf.reduce_mean(\
                tf.math.add(self.place_cell_loss, self.head_dir_loss))

            # Optimizer
            # TODO: Add weight decay to decoder layers
            self.optimizer = tf.train.RMSPropOptimizer(\
                learning_rate=learning_rate, momentum=.99, decay=1-0.0001)
            
            # Clip gradients
            gvs = self.optimizer.compute_gradients(self.loss)
            # Note: currently, last 6 ones here correspond to the grid cell layer
            # and the place/head cell layers. 
            capped_gvs = [(tf.clip_by_value(grad, -grad_clip_thresh, \
                grad_clip_thresh), var) \
                for grad, var in gvs[-6:]]


            new_gvs = gvs[:-6] + capped_gvs

            self.train_op = self.optimizer.apply_gradients(new_gvs)

            # Summary Stuff
            loss_summary = tf.summary.scalar('loss', self.loss)
            
            self.summary = tf.summary.merge_all(scope=self.name)


    def train_step(self, sess, data, place_cells, head_cells, y, z):

        trans_vel_data, strafe_trans_velocity_data, cos_vel_data, sin_vel_data, initial_pos_activity, \
        initial_dir_activity, place_cell_activity_labels, \
        head_cell_activity_labels, x_y_positions = \
            transform_input_data(data, place_cells, head_cells)

        # Doing the masking layer here, because... well... im terrible
        getrand01 = lambda: 1.0 if random.random() > .95 else 0.0        
        for i in range(self.batch_size):
            for j in range(self.max_time):
                mask = getrand01()
                y[i,j,:], z[i,j,:] = mask*y[i,j,:], mask*z[i,j,:]

        feed = {
            self.fwd_trans_velocity: trans_vel_data,
            self.strafe_trans_velocity: strafe_trans_velocity_data,
            self.cosine_angular_velocity: cos_vel_data,
            self.sine_angular_velocity: sin_vel_data,
            self.y: y,
            self.z: z, 
            self.dropout_prob: .5,
            self.place_cell_labels: place_cell_activity_labels,
            self.head_dir_labels: head_cell_activity_labels,
        }

        _, loss, summary = sess.run(
            [self.train_op, self.loss, self.summary], 
            feed_dict=feed)

        return loss, summary

    def get_grid_layer_activations(self, sess, data, place_cells, head_cells, y, z):

        trans_vel_data, strafe_trans_velocity_data, cos_vel_data, sin_vel_data, initial_pos_activity, \
        initial_dir_activity, place_cell_activity_labels, \
        head_cell_activity_labels, x_y_positions = \
            transform_input_data(data, place_cells, head_cells)

        feed = {
            self.fwd_trans_velocity: trans_vel_data,
            self.strafe_trans_velocity: strafe_trans_velocity_data,
            self.cosine_angular_velocity: cos_vel_data,
            self.sine_angular_velocity: sin_vel_data,
            self.y: y,
            self.z: z, 
            self.place_cell_labels: place_cell_activity_labels,
            self.head_dir_labels: head_cell_activity_labels,
            self.dropout_prob: 0,
        }

        grid_layer_activations = sess.run(self.grid_cell_layer, feed_dict=feed)
        hist = self._grid_activations_to_histogram(grid_layer_activations, 
            x_y_positions)

        return hist

    def _grid_activations_to_histogram(self, grid_layer_activations, 
            x_y_positions, bins=32):

        grid_acts = np.reshape(grid_layer_activations, (-1, self.grid_layer_size))
        xy_pos = np.reshape(x_y_positions, (-1, 2))
        data = zip(xy_pos, grid_acts)        

        def xy_coord_to_ind(loc):
            # Assuming a valid location
            x_coord, y_coord = loc
            low, high = 100, 800
            x, y = x_coord-low, y_coord-low
            size_of_bin = (high-low)/bins
            x_ind, y_ind = int(x/(size_of_bin+1)), int(y/(size_of_bin+1)) #wont have anything in last bin really?
            return x_ind, y_ind

        def construct_histogram(data):
            #janky avoid /0
            counts = np.ones((self.grid_layer_size, bins, bins))*1e-10
            activations = np.zeros((self.grid_layer_size, bins, bins))
            for loc, acts in data:
                x_ind, y_ind = xy_coord_to_ind(loc)
                for i, act in enumerate(acts):
                    activations[i, x_ind, y_ind] += act
                    counts[i, x_ind, y_ind] += 1
            histograms = activations/counts
            return histograms

        return construct_histogram(data)

    # TODO: similar function for head direction cells 


class VisionModule(object):
    def __init__(self, name='VisionModule', state_size=[80,80,3], max_time=100,
        N=256, M=12, learning_rate=1e-4):
        self.name = name
        self.state_size = state_size
        self.max_time = max_time
        self.N = N
        self.M = M
        self.learning_rate = learning_rate
        
        # self.graph = tf.Graph()
        # with self.graph.as_default():
        with tf.variable_scope(name):

            # Input images
            self.inputs = tf.placeholder(tf.float32, \
                [None, max_time, state_size[0], state_size[1], state_size[2]], \
                name='inputs')

            self.inputs_ = tf.reshape(self.inputs, \
                [-1, state_size[0], state_size[1], state_size[2]])

            # Conv layers
            # ReLU and Padding are defaults.
            # Same network used for actor-learner, but weights not shared.
            self.conv1 = tf.contrib.layers.conv2d(self.inputs_, 16, kernel_size=5, stride=2)
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, 32, kernel_size=5, stride=2)
            self.conv3 = tf.contrib.layers.conv2d(self.conv2, 64, kernel_size=5, stride=2)
            self.conv4 = tf.contrib.layers.conv2d(self.conv3, 128, kernel_size=5, stride=2)

            self.conv4_reshape = tf.reshape(self.conv4, [-1, 5*5*128])

            # Place and head cells
            self.pred_place_cell_logits = tf.contrib.layers.fully_connected(\
                self.conv4_reshape, N)

            self.pred_head_dir_logits = tf.contrib.layers.fully_connected(\
                self.conv4_reshape, M)

            # Construct loss
            self.place_cell_labels_1 = tf.placeholder(tf.float32, [None, max_time, N],
                name='place_cell_labels_1')
            self.head_dir_labels = tf.placeholder(tf.float32, [None, max_time, M],
                name='head_dir_labels')

            self.place_cell_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.place_cell_labels_1,
                logits=self.pred_place_cell_logits)
            self.head_dir_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.head_dir_labels,
                logits=self.pred_head_dir_logits)

            self.loss = tf.reduce_mean(\
                tf.math.add(self.place_cell_loss, self.head_dir_loss))

            # Optimizer
            self.optimizer = tf.train.RMSPropOptimizer(\
                learning_rate=self.learning_rate, momentum=.9).minimize(self.loss)

            # Output
            self.pred_place_cell = tf.math.softmax( \
                tf.reshape(self.pred_place_cell_logits, [-1, self.max_time, self.N]), 
                axis=2)

            self.pred_head_dir = tf.math.softmax(\
                tf.reshape(self.pred_head_dir_logits, [-1, self.max_time, self.M]), 
                axis=2)

            # Note: may want to use tf.stop gradient if hooking this up directly
            # to other network.

            loss_summary = tf.summary.scalar('loss', self.loss)
            self.summary = tf.summary.merge_all(scope=self.name)

            # TODO: Make summary
            # TODO: Mask outputs?

    def train_step(self, sess,data, place_cells, head_cells):
        trans_vel_data, strafe_trans_velocity_data, cos_vel_data, sin_vel_data, initial_pos_activity, \
        initial_dir_activity, place_cell_activity_labels, \
        head_cell_activity_labels, x_y_positions = \
            transform_input_data(data, place_cells, head_cells)
        
        # Todo: check correct shape of observation_data 
        observation_data = data[0]

        # Todo: need to concatenate output of this with a vision module

        feed = {
            self.inputs: observation_data,
            self.place_cell_labels_1: place_cell_activity_labels,
            self.head_dir_labels: head_cell_activity_labels,
        }

        # TODO: add tensorboard summaries
        _, loss, pred_place_cell, pred_head_dir, summary = sess.run(
            [self.optimizer, self.loss, self.pred_place_cell, 
            self.pred_head_dir, self.summary], 
            feed_dict=feed)

        return loss, pred_place_cell, pred_head_dir, summary



# class Trainer(object):
#     """
#     Class meant to run the training pipeline for the grid network with one set
#     of hyper parameters.
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
#         level_script='tests/empty_room_test', 
#         obs_types=['RGB_INTERLEAVED', 'VEL.TRANS', 'VEL.ROT', 'POS',
#                 'DISTANCE_TO_WALL', 'ANGLE_TO_WALL', 'ANGLES'],
#         config={'width': str(80), 'height': str(80)}):

#         self.env = environment.ParallelEnv(level_script, obs_types, config, num_envs)
#         self.place_cells = cells.PlaceCells(N, sigma)
#         self.head_cells = cells.HeadDirCells(M)
#         self.grid_network = GridNetwork(name="grid_network", 
#             learning_rate=learning_rate, max_time=trajectory_length) # add this later
#         self.rat = rat_trajectory_generator.RatTrajectoryGenerator(self.env, trajectory_length)

#         self.train_iterations = train_iterations
#         self.batch_size = batch_size

#         self.base_path = base_path
#         self.experiments_save_path = self.base_path + 'experiments/'
#         self.unique_exp_save_path = self.experiments_save_path \
#             + 'exp_' + unique_exp_name + '/'

#         self.restore = restore
#         self.saver = tf.train.Saver()

#     def train(self):
#         print("Trainer.train")
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())

#             if self.restore:
#                 self.saver.restore(sess, \
#                     tf.train.latest_checkpoint(self.unique_exp_save_path))

#             train_writer = tf.summary.FileWriter(self.unique_exp_save_path)
#             """
#             say N envs, N*10 trajectories / batch = 10s
#             Nenvs* 100steps/sec = 100*N steps/s
#             10 minutes to reach 1mil
#             estimated this will take 40+ hours on 16 cpus w/ min bottleneck of 
#             1s/ reset

#             to reach 100 mil (16+ hours) need 1mil = iterations*batchsize
#             """

#             for i in range(self.train_iterations):
#                 print("Train iter: ", i)
#                 sys.stdout.flush()
#                 data = self.rat.generateAboutNTrajectories(self.batch_size)
#                 loss, summary = self.grid_network.train_step(sess, data, self.place_cells, 
#                     self.head_cells)
#                 train_writer.add_summary(summary, i)
#                 print("loss: ", np.mean(loss))

#                 if i % 10 == 0:
#                     self.saver.save(sess, self.unique_exp_save_path)

#                     histograms = self.grid_network.get_grid_layer_activations(sess, 
#                         data, self.place_cells, self.head_cells)

#                     print("And we have histograms:")
#                     print(histograms.shape)

#                     np.save(self.unique_exp_save_path + 'place_cell_histograms.npy', \
#                         histograms)



# class SlurmManager(object):
#     """
#     Class meant to deal with running multiple jobs with different hyperparams 
#     using Slurm.
#     """

#     def __init__(self, slurm_array_index, base_path):
#         """
#         For now, let us assume slurm_task_id in {0, num_hyperparam_combos-1}.
#         """
#         self.slurm_array_index = slurm_array_index
#         self.base_path = base_path

#     def run(self):
#         # TODO: make a more elegant, scalable way to take in different 
#         # sets of hyperparams for different tasks. But for now:

#         """
#         hyperparams = [
#             learning_rate, 
#             batch_size, 
#             train_iterations, 
#             num_envs, 
#             sigma, 
#         ]
#         """

#         hyperparams = [
#             [1e-3, 1e-4, 1e-5],
#             [40],
#             [100000],
#             [4],
#             [3, 1, .5],
#         ]

#         combinations = list(itertools.product(*hyperparams))
#         print("Hyperparam combos: ", combinations)
#         hyperparam_selection = combinations[self.slurm_array_index]
#         print("Hyperparam selection: ", hyperparam_selection)
#         trainerParallel = Trainer(
#             base_path=self.base_path,
#             unique_exp_name='job_' + str(self.slurm_array_index),
#             learning_rate=hyperparam_selection[0],
#             batch_size=hyperparam_selection[1],
#             train_iterations=hyperparam_selection[2],
#             num_envs=hyperparam_selection[3],
#             sigma=hyperparam_selection[4])
#         trainerParallel.train()

def run(slurm_array_index, base_path):

    print("testing vision module:")
    v = VisionModule()

    # TESTING LOCALLY:
    base_path = '/mnt/hgfs/ryanprinster/data/'
    slurm_array_index = 1

    SlurmManager(slurm_array_index, base_path).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame_count', type=int, default=10000,
                      help='Number of steps to run the agent')
    parser.add_argument('--width', type=int, default=80,
                      help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=80,
                      help='Vertical size of the observations')
    parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
    parser.add_argument('--level_script', type=str, default='tests/empty_room_test',
                      help='The environment level script to load')
    parser.add_argument('--base_path', type=str, default='/om/user/prinster/lab/my_data/',
                      help='base_path')
    parser.add_argument('--num_envs', type=str, default=16,
                      help='num environments to run in parallel')
    parser.add_argument('--learning_rate', type=str, default=1e-3,
                      help='learning_rate')
    parser.add_argument('--exp_name', type=str, default='test',
                      help='exp_name')
    parser.add_argument('--slurm_array_index', type=int, default=0,
                      help='id provided by slurm for which experiment to run')

    args = parser.parse_args()
    if args.runfiles_path:
        deepmind_lab.set_runfiles_path(args.runfiles_path)
    run(args.slurm_array_index, args.base_path)
