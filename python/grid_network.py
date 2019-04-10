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
    """Basic random agent for DeepMind Lab."""
    def __init__(self, name, lstm_size, grid_layer_size, N, M):
        with tf.variable_scope(name):
            
            # Constructing inputs to the lstm layer
            self.translational_velocity = tf.placeholder(tf.float32, [None, None, 1])
            self.sine_angular_velocity = tf.placeholder(tf.float32, [None, None, 1])
            self.cosine_angular_velocity = tf.placeholder(tf.float32, [None, None, 1])

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
            self.place_cell_labels = tf.placeholder(tf.float32, [None, None, N])
            self.head_dir_labels = tf.placeholder(tf.float32, [None, None, M])

            self.place_cell_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.place_cell_labels,
                logits=self.pred_place_cell_logits)
            self.head_dir_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.head_dir_labels,
                logits=self.pred_head_dir_logits)

            self.loss = tf.math.add(self.place_cell_loss, self.head_dir_loss)


            # Optimizer
            # TODO: Add weight decay to decoder layers
            # TODO: Add gradient clipping to output of grid cell layer
            self.optimizer = tf.train.RMSPropOptimizer(self.loss, momentum=.9)


def run(width, height, level_script, frame_count):
    """Spins up an environment and runs the agent."""
    # config = {'width': str(width), 'height': str(height)}

    # env = deepmind_lab.Lab(level_script, ['RGB_INTERLEAVED'], config=config)

    learning_rate = 1e-5

    grid_network = GridNetwork(\
        name="grid_network", 
        lstm_size=128, 
        grid_layer_size=512,
        N=256, 
        M=12)
 


  


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
