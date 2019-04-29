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
"""A simple example of a random agent in deepmind_lab."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import six

from multiprocessing import Process, Queue, Pipe

import deepmind_lab

import tensorflow as tf



class GridNetwork(object):
    """Grid Network"""
    def __init__(self, name, lstm_size=128, grid_layer_size=512, N=256, M=12,
        learning_rate = 1e-5, grad_clip_thresh=1e-5, max_time=100):
        with tf.variable_scope(name):
            
            # Constructing inputs to the lstm layer
            self.translational_velocity = tf.placeholder(tf.float32, [None, max_time, 1])
            self.sine_angular_velocity = tf.placeholder(tf.float32, [None, max_time, 1])
            self.cosine_angular_velocity = tf.placeholder(tf.float32, [None, max_time, 1])

            self.lstm_input = tf.concat([
                self.translational_velocity,
                self.sine_angular_velocity,
                self.cosine_angular_velocity,],
                axis=2, name='lstm_input')

            self.place_activity = tf.placeholder(tf.float32, [None, N])
            self.head_dir_activity = tf.placeholder(tf.float32, [None, M])

            # Initial cell state and hidden state are linear transformations of 
            # place and head direction cells. Cell state and hidden state are 
            # concatenated for LSTMCell(state_is_tuple=False).

            self.lstm_init_cell_hidden_state = tf.concat([
                    tf.contrib.layers.fully_connected(self.place_activity, int(lstm_size/2), activation_fn=None),
                    tf.contrib.layers.fully_connected(self.head_dir_activity, int(lstm_size/2), activation_fn=None),
                    tf.contrib.layers.fully_connected(self.place_activity, int(lstm_size/2), activation_fn=None),
                    tf.contrib.layers.fully_connected(self.head_dir_activity, int(lstm_size/2), activation_fn=None),
                ], axis=1, name='lstm_init_cell_hidden_state')

            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=False)

            # TODO: unroll lstm for 100 time steps (paper says so)
            self.lstm_output, self.lstm_hidden_state_output = tf.nn.dynamic_rnn(
                self.lstm_cell,
                self.lstm_input,
                initial_state=self.lstm_init_cell_hidden_state,
                dtype=tf.float32)
            
            # Grid Cell layer
            self.grid_cell_layer = tf.contrib.layers.fully_connected(\
                self.lstm_output, grid_layer_size, activation_fn=None)

            self.dropout_prob = tf.placeholder(tf.float32)
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
            self.place_cell_labels = tf.placeholder(tf.float32, [None, max_time, N])
            self.head_dir_labels = tf.placeholder(tf.float32, [None, max_time, M])

            self.place_cell_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.place_cell_labels,
                logits=self.pred_place_cell_logits)
            self.head_dir_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.head_dir_labels,
                logits=self.pred_head_dir_logits)

            self.loss = tf.math.add(self.place_cell_loss, self.head_dir_loss)


            # Optimizer
            # TODO: Add weight decay to decoder layers
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=.9)
            
            # Clip gradients
            gvs = self.optimizer.compute_gradients(self.loss)
            # Note: currently, last 6 ones here correspond to the grid cell layer
            # and the place/head cell layers. 
            capped_gvs = [(tf.clip_by_value(grad, -grad_clip_thresh, grad_clip_thresh), var) for grad, var in gvs[-6:]]
            new_gvs = gvs[:-6] + capped_gvs

            self.train_op = self.optimizer.apply_gradients(new_gvs)

class PlaceCells(object):
    def __init__(self,N):
        self.N = N
        self.locations = _generate_place_cells(N)
        self.sigma = 0.01 * 32
        # Note: In DeepMind Lab there are 32 units to the meter, 
        # so we multiply any plane or position by this number.

    def _generate_place_cells(N):
        """
        Generates ground truth locations for place cells.
        TODO: do this for a given maze.
        """
        low, high = 100, 800
        place_cell_locations = np.random.uniform(low, high, (self.N, 2)) # place cell centers
        sigma = 0.01 * 32
        return place_cell_locations

    def _gaussian(x, mu, sig):
        return np.exp(-np.sum(np.power(x - mu, 2.)) / (2 * np.power(sig, 2.)))


    def get_ground_truth_activation(x):
        """For a given location, get activations of all place cells"""
        activations = []
        for mu in self.locations:
            activation = _gaussian(x, mu, self.sigma)
            activations.append(activation)
        normalized_activations = activations/np.sum(np.array(activations))
        return normalized_activations

    def get_ground_truth_activations(X):
        """ For a list of locations, get activations of all place cells"""
        many_activations = []
        for loc in X:
            many_activations.append(get_ground_truth_activation(loc))
        return many_activations

class HeadDirCells(object):
    def __init__(self, M):
        self.M = M
        


def run(width, height, level_script, frame_count):
    """Spins up an environment and runs the agent."""
    # config = {'width': str(width), 'height': str(height)}

    # env = deepmind_lab.Lab(level_script, ['RGB_INTERLEAVED'], config=config)

    learning_rate = 1e-5
    train_iterations = 2
    N=256
    M=12

    place_cells = PlaceCells(N)



    # obs_data = np.load('/mnt/hgfs/ryanprinster/test/obs_data.npy')
    # pos_data = np.load('/mnt/hgfs/ryanprinster/test/pos_data.npy')
    # dir_data = np.load('/mnt/hgfs/ryanprinster/test/dir_data.npy')

    # print("obs_data shape:", obs_data.shape)
    # print("pos_data shape:", pos_data.shape)
    # print("dir_data shape:", dir_data.shape)

    grid_network = GridNetwork(name="grid_network")
    # for i in range(train_iterations):



  


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
    parser.add_argument('--level_script', type=str, default='tests/trivial_maze',
                      help='The environment level script to load')

    args = parser.parse_args()
    if args.runfiles_path:
        deepmind_lab.set_runfiles_path(args.runfiles_path)
    run(args.width, args.height, args.level_script, args.frame_count)
